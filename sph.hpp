#pragma once

#include "math.hpp"

namespace sph
{
    void init();
    void getHandles(std::vector<sk::math::vec3>** ppparticles, std::vector<std::vector<int>>** ppparticleNeighbours);
    void update(const float currentTime);
}
