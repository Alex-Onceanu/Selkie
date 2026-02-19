// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: Fri Jan 22 20:45:07 2021
//      Author: Kiwon Um
//        Mail: kiwon.um@telecom-paris.fr
//
// Description: SPH simulator (DO NOT DISTRIBUTE!)
//
// Copyright 2021-2026 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um,
// Telecom Paris, France. The program(s) may be used and/or copied only with
// the written permission of Kiwon Um or in accordance with the terms and
// conditions stipulated in the agreement/contract under which the program(s)
// have been supplied.
// ----------------------------------------------------------------------------

#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <unistd.h>

#include "sph.hpp"

#ifndef M_PI
#define M_PI 3.141592
#endif

#include "math.hpp"

namespace sph
{
// timer
double gAppTimer = 0.0;
double gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gShowGrid = true;
bool gShowVel = false;
int gSavedCnt = 0;

const int kViewScale = 15;

// SPH Kernel function: cubic spline
class CubicSpline {
public:
  explicit CubicSpline(const double h=1.) : _dim(3)
  {
    setSmoothingLen(h);
  }
  void setSmoothingLen(const double h)
  {
    const double h2 = h * h, h3 = h2*h;
    _h = h;
    _sr = 2e0*h;
    _c[0]  = 2e0/(3e0*h);
    _c[1]  = 10e0/(7e0*M_PI*h2);
    _c[2]  = 1e0/(M_PI*h3);
    _gc[0] = _c[0]/h;
    _gc[1] = _c[1]/h;
    _gc[2] = _c[2]/h;
  }
  double smoothingLen() const { return _h; }
  double supportRadius() const { return _sr; }

  double f(const double l) const
  {
    const double q = l/_h;
    if(q<1e0) return _c[_dim-1]*(1e0 - 1.5*(q * q) + 0.75*(q*q*q));
    else if(q<2e0) return _c[_dim-1]*(0.25*(2e0-q)*(2e0-q)*(2e0-q));
    return 0;
  }
  double derivative_f(const double l) const
  {
    const double q = l/_h;
    if(q<=1e0) return _gc[_dim-1]*(-3e0*q+2.25*(q*q));
    else if(q<2e0) return -_gc[_dim-1]*0.75*(2e0-q)*(2e0-q);
    return 0;
  }

  double w(const sk::math::vec3 &rij) const { return f(rij.length()); }
  sk::math::vec3 grad_w(const sk::math::vec3 &rij) const { return grad_w(rij, rij.length() + 1e-5 * _h); }
  sk::math::vec3 grad_w(const sk::math::vec3 &rij, const double len) const
  {
    return rij*(derivative_f(len)/len);
  }

private:
  int _dim;
  double _h, _sr, _c[3], _gc[3];
};

class SphSolver {
public:
  explicit SphSolver(
    const double nu=0.08, const double h=0.5, const double density=1e3,
    const sk::math::vec3 g=sk::math::vec3(0, -9.8, 0.), const double eta=0.01, const double gamma=7.0) :
    _kernel(h), _nu(nu), _h(h), _d0(density),
    _g(g), _eta(eta), _gamma(gamma)
  {
    _dtRef = 0.0009f;
    _dt = _dtRef;
    _m0 = _d0*_h*_h*_h;
    _c = std::fabs(_g.y)/_eta;
    _k = _d0*_c*_c/_gamma;
  }

  void set_dt(double dt)
  {
    _dt = dt;
  }

  // assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
  // the size of f_width, f_height; each cell is sampled with 2x2 particles.
  void initScene(
    const int res_x, const int res_y, const int res_z, const int f_width, const int f_height, const int f_depth)
  {
    _pos.clear();

    _particlesPerCell.resize(res_x * res_y * res_z, {});
    _neighbors.resize(res_x * res_y * res_z * 4, {});

    _resX = res_x;
    _resY = res_y;
    _resZ = res_z;

    // set wall for boundary
    _l = 0.5*_h;
    _r = static_cast<double>(res_x) - 0.5*_h;
    _b = 0.5*_h;
    _t = static_cast<double>(res_y) - 0.5*_h;
    _n = 0.5*_h;
    _f = static_cast<double>(res_z) - 0.5*_h;

    // sample a fluid mass
    for(int j=0; j<f_height; ++j) {
      for(int i=0; i<f_width; ++i) {
        for(int k=0;k<f_depth; ++k) {
          _pos.push_back(sk::math::vec3(i+0.25, res_y - 1 - j+0.25, k+0.25));
          _pos.push_back(sk::math::vec3(i+0.75, res_y - 1 - j+0.25, k+0.25));
          _pos.push_back(sk::math::vec3(i+0.25, res_y - 1 - j+0.75, k+0.25));
          _pos.push_back(sk::math::vec3(i+0.75, res_y - 1 - j+0.75, k+0.25));
        }
      }
    }

    // make sure for the other particle quantities
    _vel = std::vector<sk::math::vec3>(_pos.size(), sk::math::vec3(0));
    _acc = std::vector<sk::math::vec3>(_pos.size(), sk::math::vec3(0));
    _p   = std::vector<double>(_pos.size(), 0);
    _d   = std::vector<double>(_pos.size(), 0);

    _dtRemainder = 0.0f;

    updateParticlesPerCell();
    buildNeighbor();
  }

  void update()
  {
    updateParticlesPerCell();
    buildNeighbor();

    computePressure2();
    
    _acc = std::vector<sk::math::vec3>(_pos.size(), sk::math::vec3(0));
    applyForcesAndUpdate();
    
    resolveCollision();
  }

  int particleCount() const { return _pos.size(); }
  const sk::math::vec3& position(const int i) const { return _pos[i]; }

  int resX() const { return _resX; }
  int resY() const { return _resY; }
  int resZ() const { return _resZ; }

  double equationOfState(
    const double d, const double d0,
    const double k,               // NOTE: You can use _k for k here.
    const double gamma=7.0)
  {
    return k * (powf(d / d0, gamma) - 1.f);
  }

  double get_dt_remainder()
  {
    return _dtRemainder;
  }

  void set_dt_remainder(double dt_)
  {
    _dtRemainder = dt_;
  }

  double get_dt_ref()
  {
    return _dtRef;
  }

  std::vector<sk::math::vec3>* getPosPtr()
  {
    return &_pos;
  }

  std::vector<std::vector<int>>* getNeighborsPtr()
  {
    return &_neighbors;
  }

private:
  void updateParticlesPerCell()
  {
    for(auto& v : _particlesPerCell)
    {
      v.clear();
    }

    for(int particle = 0; particle < particleCount(); particle++)
    {
      auto pos = position(particle);
      _particlesPerCell[idx1d(std::floor(pos.x), std::floor(pos.y), std::floor(pos.z))].push_back(particle);
    }
  }

  void buildNeighbor()
  {
    for(int i = 0; i < resX(); i++)
    {
      for(int j = 0; j < resY(); j++)
      {
        for(int k = 0; k < resZ(); k++)
        {
          for(const auto& particle : _particlesPerCell[idx1d(i, j, k)])
          {
            _neighbors[particle].clear();
            // for each particle, search for its neighbors
            for(int off_i = -1; off_i <= 1; off_i++)
            {
              for(int off_j = -1; off_j <= 1; off_j++)
              {
                for(int off_k = -1; off_k <= 1; off_k++)
                {
                  if(  i + off_i < 0 
                    || j + off_j < 0 
                    || k + off_k < 0
                    || i + off_i >= resX() 
                    || j + off_j >= resY()
                    || k + off_k >= resZ())
                    continue;
                  
                  for(const auto& neighbor : _particlesPerCell[idx1d(i + off_i, j + off_j, k + off_k)])
                  {
                    if((position(neighbor) - position(particle)).length() <= _kernel.supportRadius() && particle != neighbor)
                    {
                      _neighbors[particle].push_back(neighbor);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  void computePressure2()
  {
    for(int i = 0; i < particleCount(); i++)
    {
      _d[i] = _kernel.w(sk::math::vec3(0.));
      for(const auto& neighbor : _neighbors[i])
      {
        _d[i] += _kernel.w(_pos[i] - position(neighbor));
      }
      _d[i] *= _m0;
      _p[i] = std::max(0., equationOfState(_d[i], _d0, _k));
    }
  }

  void applyForcesAndUpdate()
  {
    for(int i = 0; i < particleCount(); i++)
    {
      _acc[i] += _g;

      {
        sk::math::vec3 fi = sk::math::vec3(0.);

        for(const auto& j : _neighbors[i])
        {
          auto dWij = _kernel.grad_w(_pos[i] - _pos[j]);
          fi -= dWij * (_p[i]/(_d[i]*_d[i]) + _p[j]/(_d[j]*_d[j]));

        }
        fi *= _m0;

        _acc[i] += fi;
      }

      {
        sk::math::vec3 fi = sk::math::vec3(0.);
        
        for(const auto& j : _neighbors[i])
        {
          auto uij = _vel[i] - _vel[j];
          auto xij = _pos[i] - _pos[j];
          auto dWij = _kernel.grad_w(xij);
          fi += uij * xij.dot(dWij) * (1.f / (_d[j] * (xij.dot(xij) + 0.01f * _h * _h)));
        }
        fi *= 2.f * _nu * _m0;
        
        _acc[i] += fi;
      }

      _vel[i] += _acc[i] * _dt;
      _pos[i] += _vel[i] * _dt;
    }
  }

  // simple collision detection/resolution for each particle
  void resolveCollision()
  {
    std::vector<int> need_res;
    for(int i=0; i<particleCount(); ++i) {
      if(_pos[i].x<_l || _pos[i].y<_b || _pos[i].x>_r || _pos[i].y>_t || _pos[i].z<_n || _pos[i].z>_f)
        need_res.push_back(i);
    }

    for(
      std::vector<int>::const_iterator it=need_res.begin();
      it<need_res.end();
      ++it) {
      const sk::math::vec3 p0 = _pos[*it];
      _pos[*it].x = CLAMP(_pos[*it].x, _l, _r);
      _pos[*it].y = CLAMP(_pos[*it].y, _b, _t);
      _pos[*it].z = CLAMP(_pos[*it].z, _n, _f);
      _vel[*it] = (_pos[*it] - p0)*(1./_dt);
    }

  }


  inline int idx1d(const int i, const int j, const int k) { return i + j * resX() + k * resX() * resY(); }

  const CubicSpline _kernel;

  // particle data
  std::vector<sk::math::vec3> _pos;      // position
  std::vector<sk::math::vec3> _vel;      // velocity
  std::vector<sk::math::vec3> _acc;      // acceleration
  std::vector<double>  _p;        // pressure
  std::vector<double>  _d;        // density

  std::vector< std::vector<int> > _particlesPerCell;
  std::vector< std::vector<int> > _neighbors;

  std::vector<double> _col;    // particle color; just for visualization
  std::vector<double> _vln;    // particle velocity lines; just for visualization

  // simulation
  double _dt;                     // time step
  double _dtRemainder;
  double _dtRef;

  int _resX, _resY, _resZ;             // background grid resolution

  // wall
  double _l, _r, _b, _t, _n, _f;          // wall (boundary)

  // SPH coefficients
  double _nu;                     // viscosity coefficient
  double _d0;                     // rest density
  double _h;                      // particle spacing (i.e., diameter)
  sk::math::vec3 _g;             // gravity

  double _m0;                     // rest mass
  double _k;                      // EOS coefficient

  double _eta;
  double _c;                      // speed of sound
  double _gamma;                  // EOS power factor
};

SphSolver gSolver(0.06, 0.5, 1e3, sk::math::vec3(0, -9.8, 0.), 0.01, 7.0);

void getHandles(std::vector<sk::math::vec3>** ppparticles, std::vector<std::vector<int>>** ppparticleNeighbours)
{
  if(ppparticles != nullptr) *ppparticles = gSolver.getPosPtr();
  if(ppparticleNeighbours != nullptr) *ppparticleNeighbours = gSolver.getNeighborsPtr();
}

void init()
{
  gSolver.initScene(12, 24, 3, 3, 16, 1);
}

void update(const float currentTime)
{
  double dt = currentTime - gAppTimerLastClockTime;
  gAppTimerLastClockTime = currentTime;

  double dtRef = gSolver.get_dt_ref();
  dt += gSolver.get_dt_remainder();
  int i = 0;
  while(dt >= dtRef && i < 10)
  {
    double bef = static_cast<double>(glfwGetTime());
    gSolver.update();
    double aft = static_cast<double>(glfwGetTime());
    double timeForUpdate = aft - bef;
    // std::cout << "<< Time for 1 update : " << 1000.f * timeForUpdate << "ms" << std::endl;
    dt -= dtRef;
    i++;
  }
  // std::cout << "Number of iterations : " << i << std::endl;
  gSolver.set_dt_remainder(std::max(0., dt));
}
}