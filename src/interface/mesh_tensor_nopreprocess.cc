#include "mesh_tensor.h"
#include "delaunay.h"

#include <subdivision.h>

void CopyMeshToTensor(const Mesh& m,
	torch::Tensor* ptensorV,
	torch::Tensor* ptensorF,
	int denormalize)
{
	auto& V = m.GetV();
	auto& F = m.GetF();
	auto& tensorV = *ptensorV;
	auto& tensorF = *ptensorF;

	auto int_options = torch::TensorOptions().dtype(torch::kInt32);
#ifndef USE_DOUBLE
	typedef float T;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
#else
	typedef double T;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat64);
#endif
	tensorV = torch::full({(long long)V.size(), 3}, /*value=*/0, float_options);
	tensorF = torch::full({(long long)F.size(), 3}, /*value=*/0, int_options);

	auto dataV = static_cast<T*>(tensorV.storage().data());
	auto dataF = static_cast<int*>(tensorF.storage().data());

	auto trans = m.GetTranslation();
	auto scale = m.GetScale();
	for (int i = 0; i < V.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			if (denormalize)
				dataV[i * 3 + j] = V[i][j] * scale + trans[j];
			else
				dataV[i * 3 + j] = V[i][j];
		}
	}

	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			dataF[i * 3 + j] = F[i][j];
		}
	}
}

void CopyTensorToMesh(const torch::Tensor& tensorV,
	const torch::Tensor& tensorF,
	Mesh* pm,
	int normalize)
{
	auto& m = *pm;

	auto& V = m.GetV();
	auto& F = m.GetF();

#ifndef USE_DOUBLE
	typedef float T;
#else
	typedef double T;
#endif
	auto dataV = static_cast<const T*>(tensorV.storage().data());
	auto dataF = static_cast<const int*>(tensorF.storage().data());

	int v_size = tensorV.size(0);
	int f_size = tensorF.size(0);
	V.resize(v_size);
	F.resize(f_size);

	for (int i = 0; i < V.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			V[i][j] = dataV[i * 3 + j];
		}
	}

	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			F[i][j] = dataF[i * 3 + j];
		}
	}

	if (normalize) {
		m.Normalize();
	}
}

std::vector<torch::Tensor> LoadMesh(
	const char* filename) {
	Mesh src;
	src.ReadOBJ(filename);

	torch::Tensor tensorV;
	torch::Tensor tensorF;

	CopyMeshToTensor(src, &tensorV, &tensorF, 0);

	src.Normalize();
	
	return {tensorV, tensorF};
}

std::vector<torch::Tensor> LoadCadMesh(
	const char* filename) {

	Mesh cad;
	cad.ReadOBJ(filename);

	torch::Tensor tensorV;
	torch::Tensor tensorF;
	torch::Tensor tensorE;

	CopyMeshToTensor(cad, &tensorV, &tensorF, 0);	
	auto int_options = torch::TensorOptions().dtype(torch::kInt32);
#ifndef USE_DOUBLE
	typedef float T;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat32);
#else
	typedef double T;
	auto float_options = torch::TensorOptions().dtype(torch::kFloat64);
#endif

	std::set<std::pair<int, int> > neighbors;
	auto& faces = cad.GetF();
	for (int i = 0; i < faces.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			int v1 = faces[i][j];
			int v2 = faces[i][(j + 1) % 3];
			if (v1 > v2)
				std::swap(v1, v2);
			if (v1 == v2) {
				throw std::invalid_argument( "Face edge contains same vertices.");
			}
			neighbors.insert(std::make_pair(v1, v2));
		}
	}
	tensorE = torch::full({(long long)neighbors.size(), 2}, 0, int_options);
	auto dataE = static_cast<int*>(tensorE.storage().data());
	int top = 0;
	for (auto& n : neighbors) {
		dataE[top] = n.first;
		dataE[top + 1] = n.second;
		top += 2;
	}
	return {tensorV, tensorF, tensorE};
}


void SaveMesh(const char* filename,
	const torch::Tensor& tensorV,
	const torch::Tensor& tensorF) {
	Mesh src;
	CopyTensorToMesh(tensorV, tensorF, &src);
	src.WriteOBJ(filename);
}
