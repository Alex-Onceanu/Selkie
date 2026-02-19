#pragma once

#include <iostream>
#include <map>
#include <optional>
#include <cmath>

#include "window.hpp"
#include "renderer.hpp"


/*

HELP :

R, rot
P, inpos
A, albedo
D, dimensions
E, elongation
M, roughness
C, rounding
S, scale
O, onion
K, blendStrength
B, bend
N, norm
T, twist
H, negative

*/


namespace
{
    std::map<sk::key, bool*> toggles{};
    std::optional<sk::key> mode = std::nullopt;
    std::optional<int> axis = std::nullopt; // for vec3 input you have to press [button] then X, Y, or Z to change the component
    const sk::math::vec3 intToAxis[3] = { sk::math::vec3(1., 0., 0.), sk::math::vec3(0., 1., 0.), sk::math::vec3(0., 0., 1.) };
    int digitPressed = -1;

    void keyCallback(sk::key k, bool isPressed)
    {
        if(!isPressed) return;

        if((int)k >= (int)sk::key::Zero and (int)k <= (int)sk::key::Nine)
        {
            digitPressed = (int)k - (int)sk::key::Zero;
            return;
        }

        if(mode.has_value() and (k == sk::key::X || k == sk::key::Y || k == sk::key::Z))
        {
            axis = static_cast<int>(k) - static_cast<int>(sk::key::X);
            if(mode.has_value()) std::cout << "<< mode : " << (char)(((int)mode.value() - 65) + 'A') << " & axis : " << (char)(axis.value() + 'X') << std::endl;
            return;
        }
        axis = std::nullopt;

        if(toggles.find(k) != toggles.end())
        {
            *toggles[k] = !*toggles[k];
            return;
        }
        
        mode = (k == sk::key::ESC ? std::nullopt : std::optional(k));
        // if(mode.has_value()) std::cout << "<< mode : " << (char)(((int)mode.value() - 65) + 'A') << std::endl;
    }
}

class Editor
{
private:
    const float MOUSE_DELTA_MAX = 500.f;
    std::shared_ptr<sk::Window> window;
    std::map<sk::key, std::pair<float*, sk::math::vec2>> keyToSlider{};
    std::map<sk::key, std::pair<sk::math::vec3*, sk::math::vec2>> keyToVec3{};

    // when user holds [right] click and moves the cursor, value of mode [decreases] increases
    std::optional<sk::math::vec2> anchor = std::nullopt;
    std::optional<float> initialPtrValue = std::nullopt;
    std::optional<bool> rightClicking = std::nullopt;

    // handling rotation separately
    // TODO : generalize this to a map<key, quaternion>
    std::optional<sk::key> rotationKey = std::nullopt;
    sk::math::Quaternion* pq;
    sk::math::vec2 previousMousePos{};

public:
    Editor(std::shared_ptr<sk::Window> window__)
        : window(window__)
    {
        window->setKeyEventCallback(keyCallback);
        auto wmpos = window->getMousePos();
        previousMousePos = sk::math::vec2(wmpos.first, wmpos.second);
    }

    void bind(sk::key k, float* ptr, sk::math::vec2 minmax)
    {
        keyToSlider[k] = std::make_pair(ptr, minmax);
    }

    void bind(sk::key k, sk::math::Quaternion* ptr)
    {
        rotationKey = k;
        pq = ptr;
    }

    void bind(sk::key k, bool* ptr)
    {
        toggles[k] = ptr;
    }

    void bind(sk::key k, sk::math::vec3* ptr, sk::math::vec2 minmax)
    {
        keyToVec3[k] = std::make_pair(ptr, minmax);
    }

    void update(int& pDigitPressed)
    {
        pDigitPressed = digitPressed;
        digitPressed = -1;

        if(not mode.has_value()) return; // TODO : camera rotation

        auto wmpos = window->getMousePos();
        auto mpos  = sk::math::vec2(wmpos.first, wmpos.second);
        auto click = window->getMouseClick();

        if(rotationKey.has_value() and mode.value() == rotationKey.value())
        {
            auto wmpos = window->getMousePos();
            auto mpos = sk::math::vec2(wmpos.first, wmpos.second);

            if(not axis.has_value() or (not click.first and not click.second))
            {
                previousMousePos = mpos;
                return;
            }

            float t = std::min((mpos - previousMousePos).length() / MOUSE_DELTA_MAX, 1.f);
            *pq *= sk::math::Quaternion(intToAxis[axis.value()], t * (click.first ? 1.f : -1.f));
            pq->normalized();

            previousMousePos = mpos;
            return;
        }

        float* whoToModify = nullptr;
        sk::math::vec2 limits;
        if(keyToSlider.find(mode.value()) != keyToSlider.end())
        {
            whoToModify = keyToSlider[mode.value()].first;
            limits = keyToSlider[mode.value()].second;
        }
        else if(keyToVec3.find(mode.value()) != keyToVec3.end() and axis.has_value())
        {
            float* psb[3] = { &keyToVec3[mode.value()].first->x, &keyToVec3[mode.value()].first->y, &keyToVec3[mode.value()].first->z };
            whoToModify = psb[axis.value()];
            limits = keyToVec3[mode.value()].second;
        }
        else return;

        if(click.first)
        {
            if(not rightClicking.value_or(true))
            {
                // continues clicking
                float t = std::min((mpos - anchor.value()).length() / MOUSE_DELTA_MAX, 1.f); // TODO : apply a sigmoid to this
                *whoToModify = initialPtrValue.value() + (limits.y - *whoToModify) * t;
                if(limits.y - *whoToModify < 0.1) *whoToModify = limits.y;
            }
            else
            {
                // starts clicking
                anchor = mpos;
                initialPtrValue = *whoToModify;
                rightClicking = false;
            }
        }
        else if(click.second)
        {
            if(rightClicking.value_or(false))
            {
                float t = std::min((mpos - anchor.value()).length() / MOUSE_DELTA_MAX, 1.f);
                *whoToModify = initialPtrValue.value() - (*whoToModify - limits.x) * t;
                if(*whoToModify - limits.x < 0.1) *whoToModify = limits.x;
            }
            else
            {
                anchor = mpos;
                initialPtrValue = *whoToModify;
                rightClicking = true;
            }
        }
        else
        {
            rightClicking = std::nullopt;
            anchor = std::nullopt;
            initialPtrValue = std::nullopt;
        }
    }
};
