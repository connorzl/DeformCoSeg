#include "deform.h"

#include <ceres/ceres.h>
#include <igl/point_mesh_squared_distance.h>

#include "distanceloss.h"
#include "edgeloss.h"

void Deform(Mesh& mesh, UniformGrid& grid, FT lambda, TerminateWhenSuccessCallback* callback) {
	auto& V = mesh.V;
	auto& F = mesh.F;
	
	ceres::Problem problem;

	//Move vertices
	std::vector<ceres::ResidualBlockId> v_block_ids;
	v_block_ids.reserve(V.size());
	for (int i = 0; i < V.size(); ++i) {
		ceres::CostFunction* cost_function = DistanceLoss::Create(&grid);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[i].data());
		v_block_ids.push_back(block_id);			
	}

	//Enforce rigidity
	std::vector<ceres::ResidualBlockId> edge_block_ids;
	edge_block_ids.reserve(3 * F.size());
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
			ceres::CostFunction* cost_function = EdgeLoss::Create(v, lambda);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[F[i][j]].data(), V[F[i][(j + 1) % 3]].data());
			edge_block_ids.push_back(block_id);
		}
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.num_threads = 1;
	if (callback) {
		double prev_cost = 1e30;
		options.callbacks.push_back(callback);

		while (true) {
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			if (std::abs(prev_cost - summary.final_cost) < 1e-6)
				break;
			prev_cost = summary.final_cost;
		}
	} else {
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
	}

	//V error
	ceres::Problem::EvaluateOptions v_options;
	v_options.residual_blocks = v_block_ids;
	double v_cost;
	problem.Evaluate(v_options, &v_cost, NULL, NULL, NULL);
	std::cout<<"Vertices cost: "<<v_cost<<std::endl;

	//E error
	ceres::Problem::EvaluateOptions edge_options;
	edge_options.residual_blocks = edge_block_ids;
	FT edge_cost;
	problem.Evaluate(edge_options, &edge_cost, NULL, NULL, NULL);
	std::cout<<"Rigidity cost: "<<edge_cost<<std::endl;

	FT final_cost = v_cost + edge_cost;
	std::cout<<"Final cost: "<<final_cost<<std::endl;
}

void DeformWithRot(Mesh& mesh, UniformGrid& grid, FT lambda, TerminateWhenSuccessCallback* callback) {
	auto& V = mesh.V;
	auto& F = mesh.F;
	
	ceres::Problem problem;

	//Move vertices
	std::vector<ceres::ResidualBlockId> v_block_ids;
	v_block_ids.reserve(V.size());
	for (int i = 0; i < V.size(); ++i) {
		ceres::CostFunction* cost_function = DistanceLoss::Create(&grid);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[i].data());
		v_block_ids.push_back(block_id);			
	}

	//Enforce rigidity
	std::vector<ceres::ResidualBlockId> edge_block_ids;
	edge_block_ids.reserve(3 * F.size());
	std::vector<double> rots(V.size() * 3, 0);
	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
			ceres::CostFunction* cost_function = EdgeLossWithRot::Create(v, lambda);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0,
				V[F[i][j]].data(),
				V[F[i][(j + 1) % 3]].data(),
				rots.data() + F[i][j] * 3,
				rots.data() + F[i][(j + 1) % 3] * 3
				);
			edge_block_ids.push_back(block_id);
		}
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.num_threads = 1;
	if (callback) {
		double prev_cost = 1e30;
		options.callbacks.push_back(callback);

		while (true) {
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			if (std::abs(prev_cost - summary.final_cost) < 1e-6)
				break;
			prev_cost = summary.final_cost;
		}
	} else {
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
	}
	//V error
	ceres::Problem::EvaluateOptions v_options;
	v_options.residual_blocks = v_block_ids;
	double v_cost;
	problem.Evaluate(v_options, &v_cost, NULL, NULL, NULL);
	std::cout<<"Vertices cost: "<<v_cost<<std::endl;

	//E error
	ceres::Problem::EvaluateOptions edge_options;
	edge_options.residual_blocks = edge_block_ids;
	FT edge_cost;
	problem.Evaluate(edge_options, &edge_cost, NULL, NULL, NULL);
	std::cout<<"Rigidity cost: "<<edge_cost<<std::endl;

	FT final_cost = v_cost + edge_cost;
	std::cout<<"Final cost: "<<final_cost<<std::endl;
}

void DeformSubdivision(Subdivision& sub, UniformGrid& grid, FT lambda, TerminateWhenSuccessCallback* callback) {
	auto& mesh = sub.subdivide_mesh;
	auto& V = mesh.V;
	auto& F = mesh.F;
	
	ceres::Problem problem;

	//Move vertices 
	std::vector<ceres::ResidualBlockId> v_block_ids;
	v_block_ids.reserve(V.size());
	for (int i = 0; i < V.size(); ++i) {
		ceres::CostFunction* cost_function = DistanceLoss::Create(&grid);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[i].data());
		v_block_ids.push_back(block_id);			
	}

	//Enforce rigidity
	std::vector<ceres::ResidualBlockId> edge_block_ids;
	edge_block_ids.reserve(3 * F.size());

	edge_block_ids.reserve(sub.geometry_neighbor_pairs.size());
	for (auto& p : sub.geometry_neighbor_pairs) {
		int v1 = p.first;
		int v2 = p.second;
		Vector3 v = (V[v1] - V[v2]);
		ceres::CostFunction* cost_function = AdaptiveEdgeLoss::Create(v, lambda);
		ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[v1].data(), V[v2].data());
		edge_block_ids.push_back(block_id);
	}

	for (int i = 0; i < F.size(); ++i) {
		for (int j = 0; j < 3; ++j) {
			Vector3 v = (V[F[i][j]] - V[F[i][(j + 1) % 3]]);
			ceres::CostFunction* cost_function = AdaptiveEdgeLoss::Create(v, lambda);
			ceres::ResidualBlockId block_id = problem.AddResidualBlock(cost_function, 0, V[F[i][j]].data(), V[F[i][(j + 1) % 3]].data());
			edge_block_ids.push_back(block_id);
		}
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	options.num_threads = 1;
	if (callback) {
		double prev_cost = 1e30;
		options.callbacks.push_back(callback);

		while (true) {
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			if (std::abs(prev_cost - summary.final_cost) < 1e-6)
				break;
			prev_cost = summary.final_cost;
		}
	} else {
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
	}

	//V error
	ceres::Problem::EvaluateOptions v_options;
	v_options.residual_blocks = v_block_ids;
	double v_cost;
	problem.Evaluate(v_options, &v_cost, NULL, NULL, NULL);
	std::cout<<"Vertices cost: "<<v_cost<<std::endl;

	//E error
	ceres::Problem::EvaluateOptions edge_options;
	edge_options.residual_blocks = edge_block_ids;
	FT edge_cost;
	problem.Evaluate(edge_options, &edge_cost, NULL, NULL, NULL);
	std::cout<<"Rigidity cost: "<<edge_cost<<std::endl;

	FT final_cost = v_cost + edge_cost;
	std::cout<<"Final cost: "<<final_cost<<std::endl;	
}

void ReverseDeform(Mesh& src, Mesh& tar, FT lambda) {
	Eigen::Matrix<FT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> V1(src.V.size(), 3), V2(tar.V.size(), 3);
	Eigen::MatrixXi	F1(src.F.size(), 3), F2(tar.F.size(), 3);

	for (int i = 0; i < src.V.size(); ++i)
		V1.row(i) = src.V[i];

	for (int i = 0; i < tar.V.size(); ++i)
		V2.row(i) = tar.V[i];

	for (int i = 0; i < src.F.size(); ++i)
		F1.row(i) = src.F[i];

	for (int i = 0; i < tar.F.size(); ++i)
		F2.row(i) = tar.F[i];

	TerminateWhenSuccessCallback callback;

	double prev_cost = 1e30;
	int step = 0;
	std::vector<ceres::CostFunction*> cost_function1, cost_function2;
	auto Vc = V1;
	while (true) {
		MatrixX sqrD;
		Eigen::VectorXi I;
		MatrixX C;

		igl::point_mesh_squared_distance(V2,V1,F1,sqrD,I,C);

		ceres::Problem problem;

		auto& V = V1;
		auto& F = F1;

		//Move vertices

		for (int i = 0; i < C.rows(); ++i) {
			int find = I[i];
			MatrixX weight;
			igl::barycentric_coordinates(C.row(i), V1.row(F1(find, 0)), V1.row(F1(find, 1)), V1.row(F1(find, 2)), weight);
			Vector3 w = weight.row(0);
			ceres::CostFunction* cost_function = BarycentricDistanceLoss::Create(w, V2.row(i));

			problem.AddResidualBlock(cost_function, 0,
				&V1(F1(find, 0), 0),&V1(F1(find, 1), 0),&V1(F1(find, 2), 0));
		}

		//Enforce rigidity
		for (int i = 0; i < V.rows(); ++i) {
			ceres::CostFunction* cost_function = PointRegularizerLoss::Create(1e-3, Vc.row(i));
			problem.AddResidualBlock(cost_function, 0, &(V(i,0)));
		}

		for (int i = 0; i < F.rows(); ++i) {
			for (int j = 0; j < 3; ++j) {
				Vector3 v = (Vc.row(F(i,j)) - Vc.row(F(i,(j + 1) % 3)));
				ceres::CostFunction* cost_function = EdgeLoss::Create(v, lambda);
				problem.AddResidualBlock(cost_function, 0, &V(F(i,j),0), &V(F(i,(j + 1) % 3),0));
			}
		}

		ceres::Solver::Options options;
		options.max_num_iterations = 100;
		options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
		options.minimizer_progress_to_stdout = false;
		options.num_threads = 1;

		options.callbacks.push_back(&callback);

		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);

		if (std::abs(summary.final_cost - prev_cost) < 1e-6)
			break;
		prev_cost = summary.final_cost;
		step += 1;
		if (step == 30)
			break;
	}

	printf("Step used %d\n", step);
	for (int i = 0; i < src.V.size(); ++i)
		src.V[i] = V1.row(i);
}