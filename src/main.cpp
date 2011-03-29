#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_FECrsGraph.h>
#include <Epetra_FECrsMatrix.h>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_CrsGraph.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <typeinfo.hpp>

using namespace Teuchos;
namespace po = boost::program_options;

struct MeshInfo {
  OrdinalT num_global_nodes;
  int num_dofs_per_node;
  OrdinalT num_global_rows;

  OrdinalT node_begin;
  OrdinalT node_end;
};

void try_e_fecrs(MPI_Comm & Comm, const MeshInfo & meshInfo);
void try_t_fecrs(MPI_Comm & Comm, const MeshInfo & meshInfo);

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

  try_e_fecrs(pMachine, meshInfo);
  try_t_fecrs(pMachine, meshInfo);

  stk::parallel_machine_finalize();
}

/* 2-quad element mesh with the following node numbers:
 *
 *   5-----4
 *   |     |
 *   |     |
 *   2-----1
 *   |     |
 *   |     |
 *   3-----0
 */
void build_epetra_graph(RCP<Epetra_FECrsGraph> graph, const MeshInfo & mi) {
  int nDof = mi.num_dofs_per_node;
  for (OrdinalT row_node = mi.node_begin; row_node < mi.node_end; ++row_node) {
    OrdinalT row_offset = row_node * mi.num_dofs_per_node;
    for (int row_dof = 0; row_dof < mi.num_dofs_per_node; ++row_dof) {
      const OrdinalT gRowId = row_offset + row_dof;
      std::vector<OrdinalT> gCols;
      for (int col_dof = 0; col_dof < mi.num_dofs_per_node; ++col_dof) {
        switch (row_node) {
        case 0:
        case 3:
          gCols.push_back( 0 * nDof + col_dof );
          gCols.push_back( 1 * nDof + col_dof );
          gCols.push_back( 2 * nDof + col_dof );
          gCols.push_back( 3 * nDof + col_dof );
          break;
        case 1:
        case 2:
          gCols.push_back( 0 * nDof + col_dof );
          gCols.push_back( 1 * nDof + col_dof );
          gCols.push_back( 2 * nDof + col_dof );
          gCols.push_back( 3 * nDof + col_dof );
          gCols.push_back( 4 * nDof + col_dof );
          gCols.push_back( 5 * nDof + col_dof );
          break;
        case 4:
        case 5:
          gCols.push_back( 1 * nDof + col_dof );
          gCols.push_back( 2 * nDof + col_dof );
          gCols.push_back( 4 * nDof + col_dof );
          gCols.push_back( 5 * nDof + col_dof );
          break;
        default:
          throw "oops";
        }
      }
      graph->InsertGlobalIndices(1, &gRowId, gCols.size(), &gCols[0]);
    }
  }
  graph->GlobalAssemble();
  graph->OptimizeStorage();
}

void build_tpetra_graph(RCP<GraphT> graph, const MeshInfo & mi) {
  int nDof = mi.num_dofs_per_node;
  for (OrdinalT row_node = mi.node_begin; row_node < mi.node_end; ++row_node) {
    OrdinalT row_offset = row_node * mi.num_dofs_per_node;
    for (int row_dof = 0; row_dof < mi.num_dofs_per_node; ++row_dof) {
      const OrdinalT gRowId = row_offset + row_dof;
      Array<OrdinalT> gCols;
      for (int col_dof = 0; col_dof < mi.num_dofs_per_node; ++col_dof) {
        switch (row_node) {
        case 0:
        case 3:
          gCols.push_back(0 * nDof + col_dof);
          gCols.push_back(1 * nDof + col_dof);
          gCols.push_back(2 * nDof + col_dof);
          gCols.push_back(3 * nDof + col_dof);
          break;
        case 1:
        case 2:
          gCols.push_back(0 * nDof + col_dof);
          gCols.push_back(1 * nDof + col_dof);
          gCols.push_back(2 * nDof + col_dof);
          gCols.push_back(3 * nDof + col_dof);
          gCols.push_back(4 * nDof + col_dof);
          gCols.push_back(5 * nDof + col_dof);
          break;
        case 4:
        case 5:
          gCols.push_back(1 * nDof + col_dof);
          gCols.push_back(2 * nDof + col_dof);
          gCols.push_back(4 * nDof + col_dof);
          gCols.push_back(5 * nDof + col_dof);
          break;
        default:
          throw "oops";
        }
      }
      graph->insertGlobalIndices(gRowId, gCols);
    }
  }
  graph->fillComplete();
}

void try_e_fecrs(MPI_Comm & mpiComm, const MeshInfo & mi) {
  Epetra_MpiComm epetra_mpicomm = mpiComm;
  std::vector<OrdinalT> locally_owned_row_ids;
  for (OrdinalT nid = mi.node_begin; nid < mi.node_end; ++nid)
    for (int dof = 0; dof < mi.num_dofs_per_node; ++dof) {
      OrdinalT gid = nid * mi.num_dofs_per_node + dof;
      locally_owned_row_ids.push_back(gid);
    }

  RCP<Epetra_Map> row_map = rcp(
      new Epetra_Map(mi.num_global_rows, locally_owned_row_ids.size(),
          &locally_owned_row_ids[0], 0, // 0-basded counting for local row ids
          epetra_mpicomm));
  std::cout << "row_map = " << *row_map << std::endl;

  const OrdinalT approx_cols_per_row = 0;
  RCP<Epetra_FECrsGraph> fecrs_graph = rcp(
      new Epetra_FECrsGraph(::Copy, *row_map, approx_cols_per_row));

  build_epetra_graph(fecrs_graph, mi);

  std::cout << "fecrs_graph = " << *fecrs_graph << std::endl;

  RCP<Epetra_FECrsMatrix> fecrs_matrix = rcp(
      new Epetra_FECrsMatrix(::Copy, *fecrs_graph));
  fecrs_matrix->OptimizeStorage();
  fecrs_matrix->PutScalar(0);

  std::cout << "fecrs_matrix = " << *fecrs_matrix << std::endl;
}

void try_t_fecrs(MPI_Comm & mpiComm, const MeshInfo & mi) {
  //
  // Get the default communicator and node
  //
  ParameterList nodePList;
  nodePList.set<int>("Num Threads", 2, "Number of threads to use.");
  RCP<KNodeT> knode = rcp(new KNodeT(nodePList));
  RCP<PlatformT> platform = rcp(new PlatformT(knode));
  RCP<CommT> comm = platform->getComm();
  const int myRank = comm->getRank();
  std::cout << "myRank = " << myRank << std::endl;

  RCP<const MapT> map = Tpetra::createUniformContigMapWithNode<OrdinalT,OrdinalT,KNodeT>(
      mi.num_global_rows, comm, knode);

  RCP<GraphT> graph = rcp(new GraphT(map,0));
  build_tpetra_graph(graph, mi);
  RCP<MatrixT> matrix = rcp(new MatrixT(graph));
  matrix->setAllToScalar(0.0);
  Array<OrdinalT> gCols(1);
  Array<ScalarT> vals(1);
  vals[0] = 1;
  for(OrdinalT gRow=0; gRow < mi.num_global_rows; ++gRow) {
    gCols[0] = gRow;
    matrix->sumIntoGlobalValues(gRow,gCols,vals);
  }
  matrix->fillComplete();
  std::cout << "Tpetra Matrix = " << *matrix << std::endl;
}
