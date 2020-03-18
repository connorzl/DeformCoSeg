#ifndef SHAPEDEFORM_INTERFACE_DEFORM_PARAMS_H_
#define SHAPEDEFORM_INTERFACE_DEFORM_PARAMS_H_

#include <mesh.h>
#include <torch/extension.h>

struct DeformParams
{
	DeformParams()
	: scale(1.0), trans(0, 0, 0)
	{}
	Mesh ref;
	UniformGrid grid;

	FT scale;
	Vector3 trans;

#ifndef USE_DOUBLE
	std::vector<Eigen::Vector3f> edge_offset;
#else
	std::vector<Eigen::Vector3d> edge_offset;
#endif
};

extern DeformParams params;
void InitializeDeformTemplate(
	torch::Tensor tensorV,
	torch::Tensor tensorF,
	int symmetry,
	int grid_resolution);

#endif