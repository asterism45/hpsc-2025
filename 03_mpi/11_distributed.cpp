#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

struct Body {
  double x, y, m, fx, fy;
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int N = 20;
  const int localN = N / size;

  Body ibody[localN];
  Body *jbody = (Body*)MPI_ALLOC_MEM(localN * sizeof(Body), MPI_INFO_NULL);

  srand48(rank);
  for (int i = 0; i < localN; i++) {
    double x = drand48();
    double y = drand48();
    double m = drand48();
    ibody[i] = {x, y, m, 0.0, 0.0};
    jbody[i] = ibody[i];
  }

  MPI_Datatype MPI_BODY;
  MPI_Type_contiguous(5, MPI_DOUBLE, &MPI_BODY);
  MPI_Type_commit(&MPI_BODY);

  MPI_Win win;
  MPI_Win_create(jbody, localN * sizeof(Body), sizeof(Body),
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  int send_to = (rank - 1 + size) % size;
  for (int step = 0; step < size; ++step) {
    MPI_Win_fence(0, win);

    for (int i = 0; i < localN; i++) {
      for (int j = 0; j < localN; j++) {
        double rx = ibody[i].x - jbody[j].x;
        double ry = ibody[i].y - jbody[j].y;
        double r2 = rx * rx + ry * ry;
        if (r2 > 1e-30) {
          double rinv3 = 1.0 / std::sqrt(r2 * r2 * r2);
          ibody[i].fx -= rx * jbody[j].m * rinv3;
          ibody[i].fy -= ry * jbody[j].m * rinv3;
        }
      }
    }

    MPI_Put(jbody, localN, MPI_BODY,send_to,0,localN, MPI_BODY,win);

    MPI_Win_fence(0, win);
  }

  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (r == rank) {
      for (int i = 0; i < localN; i++) {
        printf("%d %g %g\n", rank*localN + i, ibody[i].fx, ibody[i].fy);
      }
    }
  }

  MPI_Win_free(&win);
  MPI_Type_free(&MPI_BODY);
  MPI_FREE_MEM(jbody);
  MPI_Finalize();
  return 0;
}
