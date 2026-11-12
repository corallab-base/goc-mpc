#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

// VAMP collision queries behind a plain-float interface. The VAMP headers
// stay in vamp_environment.cpp, the one file built with VAMP's AVX flags.

// A rigid frame change: p_frame = rotation * p_world + translation
// (rotation row-major).
struct VampFrame {
	std::array<float, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
	std::array<float, 3> translation{0, 0, 0};
};

// A static set of spheres and boxes, held in a frame.
class VampEnvironment {
public:
	struct Box {
		std::array<float, 3> center;
		std::array<float, 3> half_extents;  // axis-aligned in the world
		float margin = 0.0f;
		std::string name;
	};
	struct Ball {
		std::array<float, 3> center;
		float radius;
		float margin = 0.0f;
		std::string name;
	};

	// Shapes given in the world, held in `frame`.
	VampEnvironment(const std::vector<Box>& boxes, const std::vector<Ball>& balls, const VampFrame& frame = {});
	~VampEnvironment();
	VampEnvironment(VampEnvironment&&) noexcept;
	VampEnvironment& operator=(VampEnvironment&&) noexcept;

	// For each query sphere (x[i], y[i], z[i], r[i]) in the frame, the name of
	// a shape it comes within that shape's margin of, or "" when clear of all.
	// Safe to call concurrently.
	std::vector<std::string> FirstHits(const std::vector<float>& x, const std::vector<float>& y,
					   const std::vector<float>& z, const std::vector<float>& r) const;

private:
	friend class VampRobot;
	struct Impl;
	std::unique_ptr<Impl> _impl;
};

// A robot compiled into VAMP's collision kernels (cricket-generated, in
// vamp_robots/): sphere forward kinematics and collision checks of its
// spheres against an environment held in the robot's base frame, plus
// self-collision between the link pairs its SRDF leaves checked.
class VampRobot {
public:
	// The robots compiled in.
	static std::vector<std::string> Names();
	// Throws for a name not in Names().
	static std::unique_ptr<VampRobot> Make(const std::string& name);
	virtual ~VampRobot() = default;

	virtual int dimension() const = 0;
	// (x, y, z, r) of every sphere at q, in the base frame.
	virtual std::vector<std::array<float, 4>> Spheres(const std::vector<float>& q) const = 0;

	struct Result {
		bool free = true;
		// When not free: sphere `sphere` within the margin of `hit`, or, with
		// `hit` empty, spheres `sphere` and `other` in self-collision.
		int sphere = -1;
		int other = -1;
		std::string hit;
	};
	// One result per configuration of `configurations` (row-major,
	// dimension() per row). Safe to call concurrently.
	virtual std::vector<Result> Check(const VampEnvironment& environment,
					  const std::vector<float>& configurations) const = 0;

protected:
	// The environment's shapes as the kernels take them.
	static const void* KernelEnvironment(const VampEnvironment& environment);
};
