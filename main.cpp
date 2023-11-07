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
    MPI_Comm neigh_comm;
    dolfinx::common::Timer timer0("MPI_Dist_graph_create_adjacent");
    {
      const std::vector<int> src
          = dolfinx::MPI::compute_graph_edges_nbx(comm, dest);
      int err = MPI_Dist_graph_create_adjacent(
          comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
          dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neigh_comm);
      dolfinx::MPI::check_error(comm, err);
    }
    timer0.stop();
    MPI_Comm_free(&neigh_comm);

    MPI_Barrier(comm);
    dolfinx::common::Timer timer1("MPI_Dist_graph_create");
    MPI_Comm neigh_comm_new;
    {
      int err = MPI_Dist_graph_create(
          comm, dest.size(), dest.data(), MPI_UNWEIGHTED, dest.size(),
          dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &neigh_comm_new);
      dolfinx::MPI::check_error(comm, err);
    }
    timer1.stop();

    // Print table with information about mesh and communicator
    std::cout << "Rank: " << rank << " Size: " << size << " N: " << n
              << std::endl;

    // Compute average number of neighbors per process
    int num_neighbors = dest.size();
    int total = 0;
    MPI_Allreduce(&num_neighbors, &total, 1, MPI_INT, MPI_SUM, comm);

    double avg = double(total) / size;
    std::cout << "Average number of neighbors: " << avg << std::endl;

    // wall, user and system time in seconds
    std::array<double, 3> elapsed = timer.elapsed();

    // Print timings
    for (int i = 0; i < size; i++)
    {
      MPI_Barrier(comm);
      if (i == rank)
        std::cout << "Rank " << rank << " - Wall time: " << elapsed[0]
                  << std::endl;
    }
  }
  MPI_Finalize();
}