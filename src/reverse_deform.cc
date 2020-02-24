#include <iostream>
#include <fstream>
#include <unordered_map>

#include <igl/point_mesh_squared_distance.h>

#include "deform.h"
#include "mesh.h"
#include "meshcover.h"
#include "uniformgrid.h"

// flags
int GRID_RESOLUTION = 64;
int MESH_RESOLUTION = 5000;

// main function
int main(int argc, char** argv) {	
	if (argc < 5) {
		printf("./deform source.obj reference.obj output.obj [GRID_RESOLUTION=64] [MESH_RESOLUTION=5000] [lambda=1] [symmetry=0]\n");
		return 0;
	}
	//Deform source to fit the reference

	Mesh src, ref, cad;
	src.ReadOBJ(argv[1]);
	ref.ReadOBJ(argv[2]);

	int symmetry = 0;
	if (argc > 7) {
		sscanf(argv[6], "%d", &symmetry);
	}

	if (symmetry)
		ref.ReflectionSymmetrize();

	//cad.ReadOBJ(argv[3]);

	//MeshCover shell;
	//shell.Cover(src, cad);

	//shell.cover.WriteOBJ("debug.obj");

	//src = shell.cover;
	//return 0;
	if (argc > 4)
		sscanf(argv[4], "%d", &GRID_RESOLUTION);

	if (argc > 5)
		sscanf(argv[5], "%d", &MESH_RESOLUTION);

	FT lambda = 1;
	if (argc > 6)
		sscanf(argv[6], "%lf", &lambda);
	printf("lambda %lf\n", lambda);
	//Get number of vertices and faces
	std::cout<<"Source:\t\t"<<"Num vertices: "<<src.V.size()<<"\tNum faces: "<<src.F.size()<<std::endl;
	std::cout<<"Reference:\t"<<"Num vertices: "<<ref.V.size()<<"\tNum faces: "<<ref.F.size()<<std::endl<<std::endl;

	{
		Mesh src_copy = src;
		Mesh ref_copy = ref;
		UniformGrid grid(GRID_RESOLUTION);
		src_copy.Normalize();
		ref_copy.ApplyTransform(src_copy);
		ref.ApplyTransform(src_copy);
		src_copy.ConstructDistanceField(grid);
		//Deform(ref_copy, grid, 20.0);

		MatrixX sqrD1, sqrD2;
		Eigen::VectorXi I1, I2;
		MatrixX C1, C2;

		Eigen::MatrixXd V1(ref_copy.V.size(), 3);
		for (int i = 0; i < ref_copy.V.size(); ++i) {
			V1.row(i) = ref_copy.V[i];
		}

		Eigen::MatrixXi F1(ref_copy.F.size(), 3);
		for (int i = 0; i < ref_copy.F.size(); ++i) {
			F1.row(i) = ref_copy.F[i];
		}

		Eigen::MatrixXd V2(src_copy.V.size(), 3);
		for (int i = 0; i < src_copy.V.size(); ++i) {
			V2.row(i) = src_copy.V[i];
		}

		Eigen::MatrixXi F2(src_copy.F.size(), 3);
		for (int i = 0; i < src_copy.F.size(); ++i) {
			F2.row(i) = src_copy.F[i];
		}

		igl::point_mesh_squared_distance(V1,V2,F2,sqrD1,I1,C1);
		igl::point_mesh_squared_distance(V2,V1,F1,sqrD2,I2,C2);

		
		std::unordered_map<long long, FT> trips;
		
		int num_entries = src_copy.V.size();
		MatrixX B = MatrixX::Zero(num_entries, 3);
		auto add_entry_A = [&](int x, int y, FT w) {
			long long key = (long long)x * (long long)num_entries + (long long)y;
			auto it = trips.find(key);
			if (it == trips.end())
				trips[key] = w;
			else
				it->second += w;
		};
		auto add_entry_B = [&](int m, const Vector3& v) {
			B.row(m) += v;
		};

		src_copy.ComputeFaceNormals();
		ref_copy.ComputeVertexNormals();

		//src_copy.WriteOBJ("../scan2cad/c1/point-src.obj", true);
		//ref_copy.WriteOBJ("../scan2cad/c1/point-ref.obj", true);
		//std::ofstream os("../scan2cad/c1/point.obj");
		int count = 0;
		for (int i = 0; i < C1.rows(); ++i) {
			auto v_ref = ref_copy.V[i];
			if (sqrt(sqrD1(i, 0)) < 0.1) {
				int find = I1[i];
				bool consistent = true;
				for (int j = 0; j < 3; ++j) {
					int vid = src_copy.F[find][j];
					Eigen::Vector3d v_s = C2.row(vid);
					if ((v_s - v_ref).norm() > 5e-2)
						consistent = false;
				}
				if (consistent) {
					MatrixX weight;
					igl::barycentric_coordinates(C1.row(i), V2.row(F2(find, 0)), V2.row(F2(find, 1)), V2.row(F2(find, 2)), weight);

					for (int j = 0; j < 3; ++j) {
						Eigen::Vector3d p0(0,0,0), p1(0,0,0);
						for (int k = 0; k < 3; ++k) {
							p0 += weight(0, k) * src_copy.V[F2(find, k)];
						}
						p1 = ref.V[i];
						Eigen::Vector3d diff = (p0 - p1);
						diff /= diff.norm();
						if (std::abs(diff.dot(src_copy.NF[find])) < 0.5) {
							continue;
						}
						if (std::abs(diff.dot(ref_copy.NV[i])) < 0.5) {
							continue;
						}
						
						for (int k = 0; k < 3; ++k) {
							add_entry_A(F2(find, j), F2(find, k), weight(0, j) * weight(0, k));
						}
						add_entry_B(F2(find, j), ref.V[i] * weight(0, j));

						//os << "v " << p0[0] << " " << p0[1] << " " << p0[2] << "\n";
						//os << "v " << p1[0] << " " << p1[1] << " " << p1[2] << "\n";
						count += 1;
					}
				}
			}
		}
		/*
		for (int i = 0; i < count; i += 1) {
			os << "l " << i * 2 + 1 << " " << i * 2 + 2 << "\n";
		}
		os.close();
		exit(0);
		*/
		
		for (int i = 0; i < src_copy.V.size(); ++i) {
			add_entry_A(i, i, 1e-6);
			add_entry_B(i, src_copy.V[i] * 1e-6);
		}

		FT regular = lambda;
		for (int i = 0; i < src_copy.F.size(); ++i) {
			for (int j = 0; j < 3; ++j) {
				int v0 = src_copy.F[i][j];
				int v1 = src_copy.F[i][(j + 1) % 3];

				double reg = regular;
				add_entry_A(v0, v0, reg);
				add_entry_A(v0, v1, -reg);
				add_entry_A(v1, v0, -reg);
				add_entry_A(v1, v1, reg);
				add_entry_B(v0, reg * (src_copy.V[v0] - src_copy.V[v1]));
				add_entry_B(v1, reg * (src_copy.V[v1] - src_copy.V[v0]));
			}
		}

		Eigen::SparseMatrix<FT> A = Eigen::SparseMatrix<FT>(num_entries, num_entries);
		std::vector<T> tripletList;
		for (auto& m : trips) {
			//printf("%d %d %lf\n", m.first / num_entries, m.first % num_entries, m.second);
			tripletList.push_back(T(m.first / num_entries, m.first % num_entries, m.second));
		}
		A.setFromTriplets(tripletList.begin(), tripletList.end());

		Eigen::SparseLU<Eigen::SparseMatrix<FT>> solver;
	    solver.analyzePattern(A);

	    solver.factorize(A);

	    //std::vector<Vector3> NV(V.size());
	    for (int j = 0; j < 3; ++j) {
	        VectorX result = solver.solve(B.col(j));

	        for (int i = 0; i < result.rows(); ++i) {
	        	src_copy.V[i][j] = result[i];
	        }
	    }
	    src_copy.WriteOBJ(argv[3]);
	}



	/*
	UniformGrid grid(GRID_RESOLUTION);
	ref.Normalize();
	src.ApplyTransform(ref);

	ref.ConstructDistanceField(grid);
	//src.HierarchicalDeform(grid);
	Deform(src, grid, lambda);
	std::cout<<"Deformed"<<std::endl;

	src.WriteOBJ(argv[3]);
	*/
	return 0;
}