// Copyright (C) 2006, 2012 Damien Hocking, KBC Advanced Technologies
// All Rights Reserved.
// This code is published under the Eclipse Public License.
//
// Authors: Damien Hocking                 KBC    2006-03-20
//        (included his original contribution into Ipopt package on 2006-03-25)
//          Andreas Waechter               IBM    2006-03-25
//           (minor changes and corrections)
//          Scott Turnberg                 CMU    2006-05-12
//           (major revision)
//           (incorporated by AW on 2006-11-11 into Ipopt package)
//
// NOTES:
// - Since Mumps 5.1.0, mumps_->nz (MUMPS_INT) is deprecated and mumps_->nnz (MUMPS_INT8) should be used.
//   For now (Mumps 5.4.0), mumps_->nz still works and has no disadvantage for us.

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
// The first header to include is the one for MPI.
// In newer ThirdParty/Mumps, mpi.h is renamed to mumps_mpi.h.
// We get informed about this by having COIN_USE_MUMPS_MPI_H defined,
// either via compiler flags or in our version of mumps_compat.h.
#include "mumps_compat.h"
#ifdef COIN_USE_MUMPS_MPI_H
#include "mumps_mpi.h"
#else
#include "mpi.h"
#endif

#include "IpMumpsSolverInterface.hpp"

#if IPOPT_SINGLE
#include "smumps_c.h"
#define MUMPS_STRUC_C SMUMPS_STRUC_C
#define mumps_c smumps_c
#else
#include "dmumps_c.h"
#define MUMPS_STRUC_C DMUMPS_STRUC_C
#define mumps_c dmumps_c
#endif

#include <cmath>
#include <cstdlib>
#include <vector>
#include <cassert>

#if !defined(IPOPT_MUMPS_NOMUTEX) && __cplusplus < 201103L
#define IPOPT_MUMPS_NOMUTEX
#endif
#ifndef IPOPT_MUMPS_NOMUTEX
#include <mutex>
/// a mutex to ensure that only one thread is running MUMPS functions at a time
static std::mutex mumps_call_mutex;
#endif

#define USE_COMM_WORLD -987654

// initialize MPI when library is loaded; finalize MPI when library is unloaded
#if defined(__GNUC__) && defined(IPOPT_MPIINIT) && !defined(MUMPS_MPI_H) && defined(HAVE_MPI_INITIALIZED)
__attribute__((constructor))
static void MPIinit(void)
{
   int mpi_initialized;
   MPI_Initialized(&mpi_initialized);
   if( !mpi_initialized )
   {
      int argc = 1;
      char** argv = NULL;
      MPI_Init(&argc, &argv);
   }
}

__attribute__((destructor))
static void MPIfini(void)
{
   int mpi_finalized;
   MPI_Finalized(&mpi_finalized);
   if(!mpi_finalized)
   {
      MPI_Finalize();
   }
}
#endif


namespace Ipopt
{
#if IPOPT_VERBOSITY > 0
static const Index dbg_verbosity = 0;
#endif

MumpsSolverInterface::MumpsSolverInterface()
{
   DBG_START_METH("MumpsSolverInterface::MumpsSolverInterface()",
                  dbg_verbosity);

   //initialize mumps
   MUMPS_STRUC_C* mumps_ = static_cast<MUMPS_STRUC_C*>(calloc(1, sizeof(MUMPS_STRUC_C)));
   mumps_->job = -1; //initialize mumps
   mumps_->par = 1; //working host for sequential version
   mumps_->sym = 2; //general symmetric matrix
   mumps_->comm_fortran = USE_COMM_WORLD;

#ifndef IPOPT_MUMPS_NOMUTEX
   const std::lock_guard<std::mutex> lock(mumps_call_mutex);
#endif

   mumps_c(mumps_);
   mumps_->icntl[2] = 0;  // global info stream
   mumps_->icntl[3] = 0;  // print level
   mumps_ptr_ = (void*) mumps_;
}

MumpsSolverInterface::~MumpsSolverInterface()
{
   DBG_START_METH("MumpsSolverInterface::~MumpsSolverInterface()",
                  dbg_verbosity);

#ifndef IPOPT_MUMPS_NOMUTEX
   const std::lock_guard<std::mutex> lock(mumps_call_mutex);
#endif

   MUMPS_STRUC_C* mumps_ = static_cast<MUMPS_STRUC_C*>(mumps_ptr_);
   mumps_->job = -2; //terminate mumps
   mumps_c(mumps_);
   delete[] mumps_->a;
   free(mumps_);
}

void MumpsSolverInterface::RegisterOptions(
   SmartPtr<RegisteredOptions> roptions
)
{
   roptions->AddLowerBoundedIntegerOption(
      "mumps_print_level",
      "Debug printing level for the linear solver MUMPS",
      0, 0,
      "0: no printing; 1: Error messages only; 2: Error, warning, and main statistic messages; 3: Error and warning messages and terse diagnostics; >=4: All information.");
   roptions->AddBoundedNumberOption(
      "mumps_pivtol",
      "Pivot tolerance for the linear solver MUMPS.",
      0.0, false,
      1.0, false,
      1e-6,
      "A smaller number pivots for sparsity, a larger number pivots for stability.");
   roptions->AddBoundedNumberOption(
      "mumps_pivtolmax",
      "Maximum pivot tolerance for the linear solver MUMPS.",
      0, false,
      1, false,
      0.1,
      "Ipopt may increase pivtol as high as pivtolmax to get a more accurate solution to the linear system.");
   roptions->AddLowerBoundedIntegerOption(
      "mumps_mem_percent",
      "Percentage increase in the estimated working space for MUMPS.",
      0,
      1000,
      "When significant extra fill-in is caused by numerical pivoting, "
      "larger values of mumps_mem_percent may help use the workspace more efficiently. "
      "On the other hand, if memory requirement are too large at the very beginning of the optimization, "
      "choosing a much smaller value for this option, such as 5, might reduce memory requirements.");
   roptions->AddBoundedIntegerOption(
      "mumps_permuting_scaling",
      "Controls permuting and scaling in MUMPS",
      0, 7,
      7,
      "This is ICNTL(6) in MUMPS.");
   roptions->AddBoundedIntegerOption(
      "mumps_pivot_order",
      "Controls pivot order in MUMPS",
      0, 7,
      7,
      "This is ICNTL(7) in MUMPS.");
   roptions->AddBoundedIntegerOption(
      "mumps_scaling",
      "Controls scaling in MUMPS",
      -2, 77,
      77,
      "This is ICNTL(8) in MUMPS.");
   roptions->AddNumberOption(
      "mumps_dep_tol",
      "Threshold to consider a pivot at zero in detection of linearly dependent constraints with MUMPS.",
      0.0,
      "This is CNTL(3) in MUMPS.", true);
   roptions->AddBoolOption(
      "mumps_dump_mtx",
      "Dump matrices to MatrixMarket files for debugging.",
      false,
      "When enabled, writes KKT matrices and permutation matrices to .mtx files.");
   roptions->AddBoolOption(
      "mumps_dump_stats",
      "Dump MUMPS statistics to log files.",
      false,
      "When enabled, writes rinfog, infog, icntl, cntl arrays to .log files.");
   roptions->AddBoolOption(
      "mumps_error_analysis",
      "Enable MUMPS error analysis.",
      false,
      "When enabled, sets ICNTL(11)=1 to compute error analysis statistics.");
}

/// give name of MUMPS with version info
std::string MumpsSolverInterface::GetName()
{
   return "MUMPS " MUMPS_VERSION;
}

bool MumpsSolverInterface::InitializeImpl(
   const OptionsList& options,
   const std::string& prefix
)
{
   Index print_level;
   options.GetIntegerValue("mumps_print_level", print_level, prefix);

   options.GetNumericValue("mumps_pivtol", pivtol_, prefix);
   if( options.GetNumericValue("mumps_pivtolmax", pivtolmax_, prefix) )
   {
      ASSERT_EXCEPTION(pivtolmax_ >= pivtol_, OPTION_INVALID, "Option \"mumps_pivtolmax\": This value must be between "
                       "mumps_pivtol and 1.");
   }
   else
   {
      pivtolmax_ = Max(pivtolmax_, pivtol_);
   }

   options.GetIntegerValue("mumps_mem_percent", mem_percent_, prefix);

   // The following option is registered by OrigIpoptNLP
   options.GetBoolValue("warm_start_same_structure", warm_start_same_structure_, prefix);

   options.GetIntegerValue("mumps_permuting_scaling", mumps_permuting_scaling_, prefix);
   options.GetIntegerValue("mumps_pivot_order", mumps_pivot_order_, prefix);
   options.GetIntegerValue("mumps_scaling", mumps_scaling_, prefix);
   options.GetNumericValue("mumps_dep_tol", mumps_dep_tol_, prefix);
   options.GetBoolValue("mumps_dump_mtx", mumps_dump_mtx_, prefix);
   options.GetBoolValue("mumps_dump_stats", mumps_dump_stats_, prefix);
   options.GetBoolValue("mumps_error_analysis", mumps_error_analysis_, prefix);

   // Reset all private data
   initialized_ = false;
   pivtol_changed_ = false;
   refactorize_ = false;
   have_symbolic_factorization_ = false;
   fact_counter_ = 0;
   solve_counter_ = 0;

   MUMPS_STRUC_C* mumps_ = static_cast<MUMPS_STRUC_C*>(mumps_ptr_);
   if( !warm_start_same_structure_ )
   {
      mumps_->n = 0;
      mumps_->nz = 0;
   }
   else
   {
      ASSERT_EXCEPTION(mumps_->n > 0 && mumps_->nz > 0, INVALID_WARMSTART,
                       "MumpsSolverInterface called with warm_start_same_structure, but the problem is solved for the first time.");
   }

   if( print_level > 0 )
   {
      mumps_->icntl[2] = 6;  // global info stream
      mumps_->icntl[3] = print_level;  // print level
   }

   return true;
}

ESymSolverStatus MumpsSolverInterface::MultiSolve(
   bool         new_matrix,
   const Index* ia,
   const Index* ja,
   Index        nrhs,
   Number*      rhs_vals,
   bool         check_NegEVals,
   Index        numberOfNegEVals
)
{
   DBG_START_METH("MumpsSolverInterface::MultiSolve", dbg_verbosity);
   DBG_ASSERT(!check_NegEVals || ProvidesInertia());
   DBG_ASSERT(initialized_);
   DBG_ASSERT(static_cast<MUMPS_STRUC_C*>(mumps_ptr_)->irn == ia);
   (void) ia;
   DBG_ASSERT(static_cast<MUMPS_STRUC_C*>(mumps_ptr_)->jcn == ja);
   (void) ja;

   if( pivtol_changed_ )
   {
      DBG_PRINT((1, "Pivot tolerance has changed.\n"));
      pivtol_changed_ = false;
      // If the pivot tolerance has been changed but the matrix is not
      // new, we have to request the values for the matrix again to do
      // the factorization again.
      if( !new_matrix )
      {
         DBG_PRINT((1, "Ask caller to call again.\n"));
         refactorize_ = true;
         return SYMSOLVER_CALL_AGAIN;
      }
   }

   // check if a factorization has to be done
   DBG_PRINT((1, "new_matrix = %d\n", new_matrix));
   if( new_matrix || refactorize_ )
   {
      ESymSolverStatus retval;
      // Do the symbolic facotrization if it hasn't been done yet
      if( !have_symbolic_factorization_ )
      {
         retval = SymbolicFactorization();
         if( retval != SYMSOLVER_SUCCESS )
         {
            return retval;
         }
         have_symbolic_factorization_ = true;
      }
      // perform the factorization
      retval = Factorization(check_NegEVals, numberOfNegEVals);
      if( retval != SYMSOLVER_SUCCESS )
      {
         DBG_PRINT((1, "FACTORIZATION FAILED!\n"));
         return retval;  // Matrix singular or error occurred
      }
      refactorize_ = false;
   }
   // do the solve
   return Solve(nrhs, rhs_vals);
}

Number* MumpsSolverInterface::GetValuesArrayPtr()
{
   MUMPS_STRUC_C* mumps_ = static_cast<MUMPS_STRUC_C*>(mumps_ptr_);
   DBG_START_METH("MumpsSolverInterface::GetValuesArrayPtr", dbg_verbosity)
   DBG_ASSERT(initialized_);
   return mumps_->a;
}

static
void dump_matrix(
   MUMPS_STRUC_C* mumps_data
)
{
//#ifdef MUMPS_DUMP_MATRIX
   // Dump the matrix
   /*for (int i = 0; i < 40; i++)
   {
      printf("%" IPOPT_INDEX_FORMAT "\n", mumps_data->icntl[i]);
   }
   for (int i = 0; i < 5; i++)
   {
      printf("%25.15e\n", mumps_data->cntl[i]);
   }
   printf("%-15d :N\n", mumps_data->n);
   printf("%-15d :NZ", mumps_data->nz);
   for (Index i = 0; i < mumps_data->nz; i++)
   {
      printf("\n%" IPOPT_INDEX_FORMAT " %" IPOPT_INDEX_FORMAT " %25.15e", mumps_data->irn[i], mumps_data->jcn[i], mumps_data->a[i]);
   }*/
   /*std::string name = "fa.mtx";
   FILE* fh = fopen(name.c_str(),"w");
   fprintf(fh, "%%%%MatrixMarket matrix coordinate real symmetric\n");
   fprintf(fh, "%d %d %d\n", mumps_data->n, mumps_data->n, mumps_data->nz);
   for (int i=0; i<mumps_data->nz; i++) {
     fprintf(fh, "%d %d %25.18e\n", mumps_data->irn[i], mumps_data->jcn[i], mumps_data->a[i]);
   }
   fclose(fh);*/
   /*
   printf("       :values");
   // Dummy RHS for now
   for (Index i = 0; i < mumps_data->n; i++)
   {
      printf("\n%25.15e", 0.);
   }
   printf("    :RHS\n");*/
//#endif

}

static
void dump_matrix_file(MUMPS_STRUC_C* mumps_data, const char* filename)
{
   FILE* fh = fopen(filename,"w");
   fprintf(fh, "%%%%MatrixMarket matrix coordinate real symmetric\n");
   fprintf(fh, "%d %d %d\n", mumps_data->n, mumps_data->n, mumps_data->nz);
   for (int i=0; i<mumps_data->nz; i++) {
     fprintf(fh, "%d %d %25.18e\n", mumps_data->irn[i], mumps_data->jcn[i], mumps_data->a[i]);
   }
   fclose(fh);
}

static
void dump_matrix_perm(MUMPS_STRUC_C* mumps_data, const char* filename)
{
   FILE* fh = fopen(filename,"w");
   fprintf(fh, "%%%%MatrixMarket matrix coordinate real symmetric\n");
   fprintf(fh, "%d %d %d\n", mumps_data->n, mumps_data->n, mumps_data->n);
   const MUMPS_INT* perm = mumps_data->sym_perm;
   if (!perm) perm = mumps_data->perm_in;
   for (int i=0; i<mumps_data->n; i++) {
     fprintf(fh, "%d %d %25.18e\n", perm[i], i+1, 1.0);
   }
   fclose(fh);
}

static
void dump_rhs_file(const Number* rhs, int n, const char* filename)
{
   FILE* fh = fopen(filename, "w");
   fprintf(fh, "%%%%MatrixMarket matrix array real general\n");
   fprintf(fh, "%d 1\n", n);
   for (int i = 0; i < n; i++) {
     fprintf(fh, "%25.18e\n", rhs[i]);
   }
   fclose(fh);
}

ESymSolverStatus MumpsSolverInterface::InitializeStructure(
   Index        dim,
   Index        nonzeros,
   const Index* ia,
   const Index* ja
)
{
   MUMPS_STRUC_C* mumps_ = static_cast<MUMPS_STRUC_C*>(mumps_ptr_);
   DBG_START_METH("MumpsSolverInterface::InitializeStructure", dbg_verbosity);

   ESymSolverStatus retval = SYMSOLVER_SUCCESS;
   if( !warm_start_same_structure_ )
   {
      mumps_->n = dim;
      mumps_->nz = nonzeros;
      delete[] mumps_->a;
      mumps_->a = NULL;

      mumps_->a = new Number[nonzeros];
      mumps_->irn = const_cast<Index*>(ia);
      mumps_->jcn = const_cast<Index*>(ja);

      // make sure we do the symbolic factorization before a real
      // factorization
      have_symbolic_factorization_ = false;
   }
   else
   {
      ASSERT_EXCEPTION(mumps_->n == dim && mumps_->nz == nonzeros, INVALID_WARMSTART,
                       "MumpsSolverInterface called with warm_start_same_structure, but the problem size has changed.");
   }

   initialized_ = true;
   return retval;
}

ESymSolverStatus MumpsSolverInterface::SymbolicFactorization()
{
   DBG_START_METH("MumpsSolverInterface::SymbolicFactorization",
                  dbg_verbosity);
   MUMPS_STRUC_C* mumps_data = static_cast<MUMPS_STRUC_C*>(mumps_ptr_);

#ifndef IPOPT_MUMPS_NOMUTEX
   const std::lock_guard<std::mutex> lock(mumps_call_mutex);
#endif

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemSymbolicFactorization().Start();
   }

   mumps_data->job = 1;      //symbolic ordering pass

   mumps_data->icntl[0] = 6;
   mumps_data->icntl[1] = 6;
   mumps_data->icntl[2] = 6;//QUIETLY!
   mumps_data->icntl[3] = 2;//4; // printing level

   mumps_data->icntl[5] = mumps_permuting_scaling_;
   mumps_data->icntl[6] = mumps_pivot_order_;
   mumps_data->icntl[7] = mumps_scaling_;
   mumps_data->icntl[9] = 0;   //no iterative refinement iterations

   if( mumps_error_analysis_ )
   {
      mumps_data->icntl[10] = 1; // error analysis
   }
   //mumps_data->icntl[11] = 1; // defines an ordering strategy for symmetric matrices
   mumps_data->icntl[12] = 1;   //avoid lapack bug, ensures proper inertia; mentioned to be very expensive in mumps manual
   mumps_data->icntl[13] = mem_percent_; //% memory to allocate over expected
   mumps_data->cntl[0] = pivtol_;  // Set pivot tolerance
   //mumps_data->cntl[1] = // iterative refinement stopping criterium

   if( mumps_dump_mtx_ )
   {
      mumps_data->write_problem[0] = 'f';
      mumps_data->write_problem[1] = '.';
      mumps_data->write_problem[2] = 'm';
      mumps_data->write_problem[3] = 't';
      mumps_data->write_problem[4] = 'x';
      mumps_data->write_problem[5] = '\0';

      std::vector<MUMPS_INT>* perm_in = new std::vector<MUMPS_INT>();
      std::vector<MUMPS_INT>& perm_in_ = *perm_in;
      int N = (mumps_data->n-2)/6;

      perm_in_.push_back(1);
      for (int i=1;i<=N;++i) {
        perm_in_.push_back(3*(i-1)+2);
        perm_in_.push_back(3*(i-1)+3);
        perm_in_.push_back(3*(i-1)+4);
        if (i==1) {
          perm_in_.push_back(3*N+2);
        }
        perm_in_.push_back(3*N+3*i+0);
        perm_in_.push_back(3*N+3*i+1);
        perm_in_.push_back(3*N+3*i+2);
      }

      perm_in_.clear();
      for (int i=1;i<=3*N+1;++i) {
        perm_in_.push_back(i);
        perm_in_.push_back(i+3*N+1);
      }


      std::vector<MUMPS_INT> source = perm_in_;
      for (int i=0;i<perm_in_.size();++i) {
        perm_in_[source[i]-1] = i+1;
      }


      printf("perm: ");
      for (int i=0;i<perm_in_.size();++i) {
        printf("%d ",perm_in_.at(i));
      }
      printf("\n");

      assert(perm_in_.size()==mumps_data->n);
      mumps_data->perm_in = perm_in_.data();

      dump_matrix_perm(mumps_data, "perm_in_kkt.mtx");

      printf("MumpsSolverInterface::SymbolicFactorization\n");
      dump_matrix_file(mumps_data, "symbolic_kkt.mtx"); // here

      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                         "hello world\n");
   }

   Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                  "Calling MUMPS-1 for symbolic factorization at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
   mumps_c(mumps_data);
   if( mumps_dump_mtx_ )
   {
      int nnz_L = mumps_data->infog[3-1];
      FILE* fh = fopen("nnz_L.mtx","w");
      fprintf(fh, "%%%%MatrixMarket matrix coordinate real symmetric\n");
      fprintf(fh, "1 1 1\n");
      fprintf(fh, "1 1 %d\n", nnz_L);
      fclose(fh);
      printf("stop\n");
      dump_matrix_perm(mumps_data, "perm_kkt.mtx");
   }
   Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                  "Done with MUMPS-1 for symbolic factorization at cpu time %10.3f (wall %10.3f).\n", CpuTime(), WallclockTime());
   Index error = mumps_data->info[0];
   const Index& mumps_permuting_scaling_used = mumps_data->infog[22];
   const Index& mumps_pivot_order_used = mumps_data->infog[6];
   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "MUMPS used permuting_scaling %" IPOPT_INDEX_FORMAT " and pivot_order %" IPOPT_INDEX_FORMAT ".\n", mumps_permuting_scaling_used, mumps_pivot_order_used);
   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "           scaling will be %" IPOPT_INDEX_FORMAT ".\n", mumps_data->icntl[7]);

   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemSymbolicFactorization().End();
   }

   //return appropriate value
   if( error == -6 )  //system is singular
   {
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) = %" IPOPT_INDEX_FORMAT " matrix is singular.\n", error);
      return SYMSOLVER_SINGULAR;
   }
   if( error < 0 )
   {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                     "Error=%" IPOPT_INDEX_FORMAT " returned from MUMPS in Factorization.\n", error);
      return SYMSOLVER_FATAL_ERROR;
   }

   return SYMSOLVER_SUCCESS;
}

ESymSolverStatus MumpsSolverInterface::Factorization(
   bool  check_NegEVals,
   Index numberOfNegEVals
)
{
   DBG_START_METH("MumpsSolverInterface::Factorization", dbg_verbosity);
   MUMPS_STRUC_C* mumps_data = static_cast<MUMPS_STRUC_C*>(mumps_ptr_);

#ifndef IPOPT_MUMPS_NOMUTEX
   const std::lock_guard<std::mutex> lock(mumps_call_mutex);
#endif

   mumps_data->job = 2;  //numerical factorization

   if( mumps_dump_mtx_ )
   {
      printf("MumpsSolverInterface::Factorization\n");
      char buffer[64];
      sprintf(buffer, "numeric_kkt_it%06d.mtx", fact_counter_);
      dump_matrix_file(mumps_data, buffer);
   }
   solve_counter_ = 0;
   fact_counter_++;

   dump_matrix(mumps_data);
   Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                  "Calling MUMPS-2 for numerical factorization.\n");
   mumps_c(mumps_data);
   Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                  "Done with MUMPS-2 for numerical factorization.\n");
   Index error = mumps_data->info[0];

   //Check for errors
   if( error == -8 || error == -9 )  //not enough memory
   {
      const Index trycount_max = 20;
      for( int trycount = 0; trycount < trycount_max; trycount++ )
      {
         Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                        "MUMPS returned INFO(1) = %" IPOPT_INDEX_FORMAT " and requires more memory, reallocating.  Attempt %d\n", error, trycount + 1);
         MUMPS_INT old_mem_percent = mumps_data->icntl[13];
         ComputeMemIncrease(mumps_data->icntl[13], 2.0 * (Number)old_mem_percent, MUMPS_INT(0), "percent extra working space for MUMPS");
         Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                        "  Increasing icntl[13] from %" IPOPT_INDEX_FORMAT " to %" IPOPT_INDEX_FORMAT ".\n", old_mem_percent, mumps_data->icntl[13]);

         dump_matrix(mumps_data);
         Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                        "Calling MUMPS-2 (repeated) for numerical factorization.\n");
         mumps_c(mumps_data);
         Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                        "Done with MUMPS-2 (repeated) for numerical factorization.\n");
         error = mumps_data->info[0];
         if( error != -8 && error != -9 )
         {
            break;
         }
      }
      if( error == -8 || error == -9 )
      {
         Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                        "MUMPS was not able to obtain enough memory.\n");
         return SYMSOLVER_FATAL_ERROR;
      }
   }

   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "Number of doubles for MUMPS to hold factorization (INFO(9)) = %" IPOPT_INDEX_FORMAT "\n", mumps_data->info[8]);
   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "Number of integers for MUMPS to hold factorization (INFO(10)) = %" IPOPT_INDEX_FORMAT "\n", mumps_data->info[9]);

   if( error == -10 )  //system is singular
   {
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) = %" IPOPT_INDEX_FORMAT " matrix is singular.\n", error);
      return SYMSOLVER_SINGULAR;
   }

   negevals_ = mumps_data->infog[11];

   if( error == -13 )
   {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) =%" IPOPT_INDEX_FORMAT " - out of memory when trying to allocate %" IPOPT_INDEX_FORMAT " %s.\nIn some cases it helps to decrease the value of the option \"mumps_mem_percent\".\n",
                     error, mumps_data->info[1] < 0 ? -mumps_data->info[1] : mumps_data->info[1],
                     mumps_data->info[1] < 0 ? "MB" : "bytes");
      return SYMSOLVER_FATAL_ERROR;
   }
   if( error < 0 )  //some other error
   {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) =%" IPOPT_INDEX_FORMAT " MUMPS failure.\n", error);
      return SYMSOLVER_FATAL_ERROR;
   }

   if( check_NegEVals && (numberOfNegEVals != negevals_) )
   {
      Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                     "In MumpsSolverInterface::Factorization: negevals_ = %" IPOPT_INDEX_FORMAT ", but numberOfNegEVals = %" IPOPT_INDEX_FORMAT "\n", negevals_,
                     numberOfNegEVals);
      return SYMSOLVER_WRONG_INERTIA;
   }

   return SYMSOLVER_SUCCESS;
}

ESymSolverStatus MumpsSolverInterface::Solve(
   Index   nrhs,
   Number* rhs_vals
)
{
   DBG_START_METH("MumpsSolverInterface::Solve", dbg_verbosity);
   MUMPS_STRUC_C* mumps_data = static_cast<MUMPS_STRUC_C*>(mumps_ptr_);

#ifndef IPOPT_MUMPS_NOMUTEX
   const std::lock_guard<std::mutex> lock(mumps_call_mutex);
#endif

   ESymSolverStatus retval = SYMSOLVER_SUCCESS;
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemBackSolve().Start();
   }
   for( Index i = 0; i < nrhs; i++ )
   {
      Index offset = i * mumps_data->n;
      mumps_data->rhs = &(rhs_vals[offset]);
      mumps_data->job = 3;  //solve
      if( mumps_dump_mtx_ )
      {
         char buffer[64];
         sprintf(buffer, "rhs_it%06d_%06d.mtx", fact_counter_ - 1, solve_counter_++);
         dump_rhs_file(mumps_data->rhs, mumps_data->n, buffer);
      }
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "Calling MUMPS-3 for solve.\n");
      mumps_c(mumps_data);
      Jnlst().Printf(J_MOREDETAILED, J_LINEAR_ALGEBRA,
                     "Done with MUMPS-3 for solve.\n");
      Index error = mumps_data->info[0];
      if( error < 0 )
      {
         Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                        "Error=%" IPOPT_INDEX_FORMAT " returned from MUMPS in Solve.\n", error);
         retval = SYMSOLVER_FATAL_ERROR;
      }
   }
   if( mumps_dump_stats_ )
   {
      int it = fact_counter_ - 1;
      int sol = solve_counter_ - 1;  // already incremented above
      bool write_header = false;
      FILE* fh = fopen("mumps_stats.csv", "r");
      if (fh == NULL) {
        write_header = true;
      } else {
        fclose(fh);
      }
      fh = fopen("mumps_stats.csv", "a");
      if (write_header) {
        fprintf(fh, "iter,solve");
        for (int i=0;i<40;++i) fprintf(fh, ",INFOG(%d)", i+1);
        for (int i=0;i<20;++i) fprintf(fh, ",RINFOG(%d)", i+1);
        for (int i=0;i<40;++i) fprintf(fh, ",ICNTL(%d)", i+1);
        for (int i=0;i<15;++i) fprintf(fh, ",CNTL(%d)", i+1);
        fprintf(fh, "\n");
      }
      fprintf(fh, "%d,%d", it, sol);
      for (int i=0;i<40;++i) fprintf(fh, ",%d", mumps_data->infog[i]);
      for (int i=0;i<20;++i) fprintf(fh, ",%e", mumps_data->rinfog[i]);
      for (int i=0;i<40;++i) fprintf(fh, ",%d", mumps_data->icntl[i]);
      for (int i=0;i<15;++i) fprintf(fh, ",%e", mumps_data->cntl[i]);
      fprintf(fh, "\n");
      fclose(fh);
   }
   if( HaveIpData() )
   {
      IpData().TimingStats().LinearSystemBackSolve().End();
   }
   return retval;
}

Index MumpsSolverInterface::NumberOfNegEVals() const
{
   DBG_START_METH("MumpsSolverInterface::NumberOfNegEVals", dbg_verbosity);
   DBG_ASSERT(negevals_ >= 0);
   return negevals_;
}

bool MumpsSolverInterface::IncreaseQuality()
{
   DBG_START_METH("MumpsTSolverInterface::IncreaseQuality", dbg_verbosity);
   if( pivtol_ == pivtolmax_ )
   {
      return false;
   }
   pivtol_changed_ = true;

   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "Increasing pivot tolerance for MUMPS from %7.2e ", pivtol_);

   //this is a more aggressive update then MA27
   //ToDo this should be tuned
   pivtol_ = Min(pivtolmax_, std::pow(pivtol_, Number(0.5)));
   Jnlst().Printf(J_DETAILED, J_LINEAR_ALGEBRA,
                  "to %7.2e.\n", pivtol_);
   return true;
}

bool MumpsSolverInterface::ProvidesDegeneracyDetection() const
{
   return true;
}

ESymSolverStatus MumpsSolverInterface::DetermineDependentRows(
   const Index*      /*ia*/,
   const Index*      /*ja*/,
   std::list<Index>& c_deps
)
{
   DBG_START_METH("MumpsSolverInterface::DetermineDependentRows",
                  dbg_verbosity);
   MUMPS_STRUC_C* mumps_data = static_cast<MUMPS_STRUC_C*>(mumps_ptr_);

   c_deps.clear();

   ESymSolverStatus retval;
   // Do the symbolic facotrization if it hasn't been done yet
   if( !have_symbolic_factorization_ )
   {
      const Index mumps_permuting_scaling_orig = mumps_permuting_scaling_;
      const Index mumps_scaling_orig = mumps_scaling_;
      mumps_permuting_scaling_ = 0;
      mumps_scaling_ = 6;
      retval = SymbolicFactorization();
      mumps_permuting_scaling_ = mumps_permuting_scaling_orig;
      mumps_scaling_ = mumps_scaling_orig;
      if( retval != SYMSOLVER_SUCCESS )
      {
         return retval;
      }
      have_symbolic_factorization_ = true;
   }
   // perform the factorization, in order to find dependent rows/columns

#ifndef IPOPT_MUMPS_NOMUTEX
   const std::lock_guard<std::mutex> lock(mumps_call_mutex);
#endif

   //Set flags to ask MUMPS for checking linearly dependent rows
   mumps_data->icntl[23] = 1;
   mumps_data->cntl[2] = mumps_dep_tol_;
   mumps_data->job = 2;   //numerical factorization

   if( mumps_dump_mtx_ )
   {
      printf("MumpsSolverInterface::DetermineDependentRows\n");
      dump_matrix_file(mumps_data, "numeric_dd_kkt.mtx"); // here
   }
   dump_matrix(mumps_data);
   mumps_c(mumps_data);
   Index error = mumps_data->info[0];

   //Check for errors
   if( error == -8 || error == -9 )  //not enough memory
   {
      const Index trycount_max = 20;
      for( int trycount = 0; trycount < trycount_max; trycount++ )
      {
         Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                        "MUMPS returned INFO(1) = %" IPOPT_INDEX_FORMAT " and requires more memory, reallocating.  Attempt %d\n", error, trycount + 1);
         MUMPS_INT old_mem_percent = mumps_data->icntl[13];
         ComputeMemIncrease(mumps_data->icntl[13], 2.0 * (Number)old_mem_percent, MUMPS_INT(0), "percent extra working space for MUMPS");
         Jnlst().Printf(J_WARNING, J_LINEAR_ALGEBRA,
                        "  Increasing icntl[13] from %" IPOPT_INDEX_FORMAT " to %" IPOPT_INDEX_FORMAT ".\n", old_mem_percent, mumps_data->icntl[13]);

         dump_matrix(mumps_data);
         mumps_c(mumps_data);
         error = mumps_data->info[0];
         if( error != -8 && error != -9 )
         {
            break;
         }
      }
      if( error == -8 || error == -9 )
      {
         Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                        "MUMPS was not able to obtain enough memory.\n");
         // Reset flags
         mumps_data->icntl[23] = 0;
         return SYMSOLVER_FATAL_ERROR;
      }
   }

   // Reset flags
   mumps_data->icntl[23] = 0;

   if( error < 0 )  //some other error
   {
      Jnlst().Printf(J_ERROR, J_LINEAR_ALGEBRA,
                     "MUMPS returned INFO(1) =%" IPOPT_INDEX_FORMAT " MUMPS failure.\n", error);
      return SYMSOLVER_FATAL_ERROR;
   }

   const Index n_deps = mumps_data->infog[27];
   for( Index i = 0; i < n_deps; i++ )
   {
      c_deps.push_back(mumps_data->pivnul_list[i] - 1);
   }

   return SYMSOLVER_SUCCESS;
}

}  //end Ipopt namespace

