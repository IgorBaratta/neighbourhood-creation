#include <dolfinx.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/timing.h>
#include <dolfinx/mesh/generation.h>

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm = MPI_COMM_WORLD;

    // get number of neighbors from command line
    int rank = dolfinx::MPI::rank(comm);
    int size = dolfinx::MPI::size(comm);

    // Find power of 3 closest to size
    std::size_t n = 1;
    while (n * n * n < std::size_t(100 * size))
      n++;

    const std::array<std::array<double, 3>, 2> p = {{{0, 0, 0}, {1, 1, 1}}};
    dolfinx::mesh::CellType cell_type = dolfinx::mesh::CellType::hexahedron;
    std::array<std::size_t, 3> N = {n, n, n};

    // create a mesh
    auto mesh = dolfinx::mesh::create_box(comm, p, N, cell_type);

    // get index map
    std::shared_ptr<const common::IndexMap> map = mesh.topology()->index_map(0);

    const std::vector<int>& dest = map->dest();

    MPI_Barrier(comm);
    MPI_Comm comm_dist_graph_adjacent;
    dolfinx::common::Timer timer0("xx MPI_Dist_graph_create_adjacent");
    {
      const std::vector<int> src
          = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
      int err = MPI_Dist_graph_create_adjacent(
          comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
          dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false,
          &comm_dist_graph_adjacent);
      dolfinx::MPI::check_error(comm, err);
    }
    timer0.stop();

    MPI_Barrier(comm);
    std::vector<int> src{rank};
    std::vector<int> degrees{dest.size()};
    dolfinx::common::Timer timer1("xx MPI_Dist_graph_create");
    MPI_Comm comm_dist_graph;
    {
      int err = MPI_Dist_graph_create(
          MPI_COMM_WORLD, src.size(), src.data(), degrees.data(), dest.data(),
          MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &comm_dist_graph);
      dolfinx::MPI::check_error(comm, err);
    }
    timer1.stop();

    // Check whether the two communicators are the same
    int same = 0;
    MPI_Comm_compare(comm_dist_graph, comm_dist_graph_adjacent, &same);
    if (same == MPI_UNEQUAL)
    {
      std::cout << "Communicators are not the same" << std::endl;
      return 1;
    }

    // Free communicators
    MPI_Comm_free(&comm_dist_graph);
    MPI_Comm_free(&comm_dist_graph_adjacent);

    // list timings
    dolfinx::list_timings(comm, {dolfinx::TimingType::wall},
                          dolfinx::Table::Reduction::max);
  }
  MPI_Finalize();
}