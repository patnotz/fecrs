#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <stk_util/parallel/Parallel.hpp>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_FECrsGraph.h>
#include <Epetra_FECrsMatrix.h>
#include <Teuchos_RCP.hpp>

using namespace Teuchos;
namespace po = boost::program_options;

typedef int IdType;

struct MeshInfo {
  IdType num_global_nodes;
  int num_dofs_per_node;
  IdType num_global_rows;

  IdType node_begin;
  IdType node_end;
};

void try_fecrs(Epetra_MpiComm & mpiComm, const MeshInfo & meshInfo);

int main(int argc, char * argv[]) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message");

  // Parse the command line options
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Handle help requests
  if (vm.count("help")) {
    std::cerr << desc << "\n";
    return 1;
  }

  stk::ParallelMachine pMachine = stk::parallel_machine_init(&argc, &argv);
  const int pSize = stk::parallel_machine_size(pMachine);
  const int pRank = stk::parallel_machine_rank(pMachine);

  MeshInfo meshInfo;
  meshInfo.num_global_nodes = 6;
  meshInfo.num_dofs_per_node = 1;
  meshInfo.num_global_rows = meshInfo.num_global_nodes
      * meshInfo.num_dofs_per_node;

  meshInfo.node_begin = int(meshInfo.num_global_nodes * (float(pRank) / pSize));
  meshInfo.node_end = int(
      meshInfo.num_global_nodes * (float(pRank + 1) / pSize));
  std::cout << "Hello from rank " << pRank << " of " << pSize << " and nodes ["
      << meshInfo.node_begin << "," << meshInfo.node_end << ")" << std::endl;

  Epetra_MpiComm mpiComm = pMachine;
  try_fecrs(mpiComm, meshInfo);

  stk::parallel_machine_finalize();
}

void try_fecrs(Epetra_MpiComm & mpiComm, const MeshInfo & mi) {

  std::vector<IdType> locally_owned_row_ids;
  for (IdType nid = mi.node_begin; nid < mi.node_end; ++nid)
    for (int dof = 0; dof < mi.num_dofs_per_node; ++dof) {
      IdType gid = nid * mi.num_dofs_per_node + dof;
      locally_owned_row_ids.push_back(gid);
    }

  RCP<Epetra_Map> row_map = rcp(
      new Epetra_Map(mi.num_global_rows, locally_owned_row_ids.size(),
          &locally_owned_row_ids[0], 0, // 0-basded counting for local row ids
          mpiComm));
  std::cout << "row_map = " << *row_map << std::endl;

  const IdType approx_cols_per_row = 0;
  RCP<Epetra_FECrsGraph> fecrs_graph = rcp(
      new Epetra_FECrsGraph(::Copy, *row_map, approx_cols_per_row));
  for (IdType row_node = mi.node_begin; row_node < mi.node_end; ++row_node) {
    IdType row_offset = row_node * mi.num_dofs_per_node;
    for (int row_dof = 0; row_dof < mi.num_dofs_per_node; ++row_dof) {
      const IdType gRowId = row_offset + row_dof;
      for (int col_dof = 0; col_dof < mi.num_dofs_per_node; ++col_dof) {
        IdType col_node = -1;
        IdType gColId = -1;
        switch (row_node) {
        case 0:
        case 3:
          col_node = 0;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 1;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 2;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 3;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          break;
        case 1:
        case 2:
          col_node = 0;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 1;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 2;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 3;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 4;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 5;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          break;
        case 4:
        case 5:
          col_node = 1;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 2;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 4;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          col_node = 5;
          gColId = col_node * mi.num_dofs_per_node + col_dof;
          fecrs_graph->InsertGlobalIndices(1, &gRowId, 1, &gColId);
          break;
        default:
          throw "oops";
        }
      }
    }
  }
  fecrs_graph->GlobalAssemble();
  fecrs_graph->OptimizeStorage();
  std::cout << "fecrs_graph = " << *fecrs_graph << std::endl;

  RCP<Epetra_FECrsMatrix> fecrs_matrix = rcp(
      new Epetra_FECrsMatrix(::Copy, *fecrs_graph));
  fecrs_matrix->OptimizeStorage();
  fecrs_matrix->PutScalar(0);

  std::cout << "fecrs_matrix = " << *fecrs_matrix << std::endl;
}
