// Copyright (C) 2004, International BusinDess Machines and others.
// All Rights Reserved.
// This code is published under the Common Public License.
//
// $Id$
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#include "IpAugRestoSystemSolver.hpp"
#include "IpCompoundMatrix.hpp"
#include "IpCompoundSymMatrix.hpp"
#include "IpCompoundVector.hpp"
#include "IpSumSymMatrix.hpp"
#include "IpDiagMatrix.hpp"

namespace Ipopt
{
  static const Index dbg_verbosity = 0;

  AugRestoSystemSolver::AugRestoSystemSolver(AugSystemSolver& orig_aug_solver,
      bool skip_orig_aug_solver_init)
      :
      AugSystemSolver(),
      orig_aug_solver_(&orig_aug_solver),
      skip_orig_aug_solver_init_(skip_orig_aug_solver_init),
      neg_omega_c_plus_D_c_cache_(1),
      neg_omega_d_plus_D_d_cache_(1),
      sigma_tilde_n_c_inv_cache_(1),
      sigma_tilde_p_c_inv_cache_(1),
      sigma_tilde_n_d_inv_cache_(1),
      sigma_tilde_p_d_inv_cache_(1),
      d_x_plus_wr_d_cache_(1),
      rhs_cR_cache_(1),
      rhs_dR_cache_(1)
  {
    DBG_START_METH("AugRestoSystemSolver::AugRestoSystemSolver()",dbg_verbosity);
  }


  AugRestoSystemSolver::~AugRestoSystemSolver()
  {
    DBG_START_METH("AugRestoSystemSolver::~AugRestoSystemSolver()",dbg_verbosity);
  }


  bool AugRestoSystemSolver::InitializeImpl(const OptionsList& options,
      const std::string& prefix)
  {
    bool retval = true;
    if (!skip_orig_aug_solver_init_) {
      retval = orig_aug_solver_->Initialize(Jnlst(), IpNLP(), IpData(), IpCq(),
                                            options, prefix);
    }

    return retval;
  }

  SymLinearSolver::ESolveStatus
  AugRestoSystemSolver::Solve(const SymMatrix* W,
                              const Vector* D_x,
                              double delta_x,
                              const Vector* D_s,
                              double delta_s,
                              const Matrix* J_c,
                              const Vector* D_c,
                              double delta_c,
                              const Matrix* J_d,
                              const Vector* D_d,
                              double delta_d,
                              const Vector& rhs_x,
                              const Vector& rhs_s,
                              const Vector& rhs_c,
                              const Vector& rhs_d,
                              Vector& sol_x,
                              Vector& sol_s,
                              Vector& sol_c,
                              Vector& sol_d,
                              bool check_NegEVals,
                              Index numberOfNegEVals)
  {
    DBG_START_METH("AugRestoSystemSolver::Solve",dbg_verbosity);
    DBG_ASSERT(J_c && J_d); // should pass these by ref

    // Remember, W and the D's may be NULL!

    SmartPtr<const CompoundSymMatrix> CW =
      dynamic_cast<const CompoundSymMatrix*>(W);

    SmartPtr<const CompoundVector> CD_x =
      dynamic_cast<const CompoundVector*>(D_x);

    SmartPtr<const CompoundMatrix> CJ_c =
      dynamic_cast<const CompoundMatrix*>(J_c);
    DBG_ASSERT(IsValid(CJ_c));

    SmartPtr<const CompoundMatrix> CJ_d =
      dynamic_cast<const CompoundMatrix*>(J_d);
    DBG_ASSERT(IsValid(CJ_d));

    SmartPtr<const CompoundVector> Crhs_x =
      dynamic_cast<const CompoundVector*>(&rhs_x);
    DBG_ASSERT(IsValid(Crhs_x));

    SmartPtr<CompoundVector> Csol_x = dynamic_cast<CompoundVector*>(&sol_x);
    DBG_ASSERT(IsValid(Csol_x));

    // Get the Sigma inverses
    SmartPtr<const Vector> sigma_n_c;
    SmartPtr<const Vector> sigma_p_c;
    SmartPtr<const Vector> sigma_n_d;
    SmartPtr<const Vector> sigma_p_d;

    if (IsValid(CD_x)) {
      sigma_n_c = CD_x->GetComp(1);
      sigma_p_c = CD_x->GetComp(2);
      sigma_n_d = CD_x->GetComp(3);
      sigma_p_d = CD_x->GetComp(4);
    }

    SmartPtr<const Vector> sigma_tilde_n_c_inv =
      Sigma_tilde_n_c_inv(sigma_n_c, delta_x, *Crhs_x->GetComp(1));
    SmartPtr<const Vector> sigma_tilde_p_c_inv =
      Sigma_tilde_p_c_inv(sigma_p_c, delta_x, *Crhs_x->GetComp(2));
    SmartPtr<const Vector> sigma_tilde_n_d_inv =
      Sigma_tilde_n_d_inv(sigma_n_d, delta_x, *Crhs_x->GetComp(3));
    SmartPtr<const Vector> sigma_tilde_p_d_inv =
      Sigma_tilde_p_d_inv(sigma_p_d, delta_x, *Crhs_x->GetComp(4));

    // Pull out the expansion matrices for d
    SmartPtr<const Matrix> pd_l = CJ_d->GetComp(0,3);
    SmartPtr<const Matrix> neg_pd_u = CJ_d->GetComp(0,4);

    // Now map the correct entries into the Solve method
    // pull out the parts of the hessian h_orig + diag
    DBG_PRINT_MATRIX(2, "CW", *CW);
    SmartPtr<const SumSymMatrix> WR_sum =
      dynamic_cast<const SumSymMatrix*>(GetRawPtr(CW->GetComp(0,0)));
    Number factor;
    SmartPtr<const SymMatrix> h_orig;
    WR_sum->GetTerm(0, factor, h_orig);
    DBG_ASSERT(factor == 1.0);
    SmartPtr<const SymMatrix> eta_DR;
    WR_sum->GetTerm(1, factor, eta_DR);
    SmartPtr<const Vector> wr_d =
      dynamic_cast<const DiagMatrix*>(GetRawPtr(eta_DR))->GetDiag();
    DBG_ASSERT(IsValid(wr_d));

    SmartPtr<const Vector> D_xR;
    if (IsValid(CD_x)) {
      D_xR = D_x_plus_wr_d(CD_x->GetComp(0), factor, *wr_d);
    }
    else {
      D_xR = D_x_plus_wr_d(NULL, factor, *wr_d);
    }

    Number delta_xR = delta_x;
    SmartPtr<const Vector> D_sR = D_s;
    Number delta_sR = delta_s;
    SmartPtr<const Matrix> J_cR = CJ_c->GetComp(0,0);
    SmartPtr<const Vector> D_cR =
      Neg_Omega_c_plus_D_c(sigma_tilde_n_c_inv, sigma_tilde_p_c_inv,
                           D_c, rhs_c);
    DBG_PRINT((2,"D_cR tag = %d\n",D_cR->GetTag()));
    Number delta_cR = delta_c;
    SmartPtr<const Matrix> J_dR = CJ_d->GetComp(0,0);
    SmartPtr<const Vector> D_dR =
      Neg_Omega_d_plus_D_d(*pd_l, sigma_tilde_n_d_inv, *neg_pd_u,
                           sigma_tilde_p_d_inv, D_d, rhs_d);
    Number delta_dR = delta_d;
    SmartPtr<const Vector> rhs_xR = Crhs_x->GetComp(0);
    SmartPtr<const Vector> rhs_sR = &rhs_s;
    SmartPtr<const Vector> rhs_cR = Rhs_cR(rhs_c, sigma_tilde_n_c_inv,
                                           *Crhs_x->GetComp(1),
                                           sigma_tilde_p_c_inv,
                                           *Crhs_x->GetComp(2));
    SmartPtr<const Vector> rhs_dR = Rhs_dR(rhs_d, sigma_tilde_n_d_inv,
                                           *Crhs_x->GetComp(3), *pd_l,
                                           sigma_tilde_p_d_inv,
                                           *Crhs_x->GetComp(4), *neg_pd_u);
    SmartPtr<Vector> sol_xR = Csol_x->GetCompNonConst(0);
    Vector& sol_sR = sol_s;
    Vector& sol_cR = sol_c;
    Vector& sol_dR = sol_d;

    SymLinearSolver::ESolveStatus
    status = orig_aug_solver_->Solve(GetRawPtr(h_orig),
                                     GetRawPtr(D_xR), delta_xR,
                                     GetRawPtr(D_sR), delta_sR,
                                     GetRawPtr(J_cR), GetRawPtr(D_cR),
                                     delta_cR,
                                     GetRawPtr(J_dR), GetRawPtr(D_dR),
                                     delta_dR,
                                     *rhs_xR, *rhs_sR, *rhs_cR, *rhs_dR,
                                     *sol_xR, sol_sR, sol_cR, sol_dR,
                                     check_NegEVals,
                                     numberOfNegEVals);

    if (status == SymLinearSolver::S_SUCCESS) {
      // Now back out the solutions for the n and p variables
      SmartPtr<Vector> sol_n_c = Csol_x->GetCompNonConst(1);
      sol_n_c->Set(0.0);
      if (IsValid(sigma_tilde_n_c_inv)) {
        sol_n_c->Copy(*Crhs_x->GetComp(1));
        sol_n_c->Axpy(-1.0, sol_cR);
        sol_n_c->ElementWiseMultiply(*sigma_tilde_n_c_inv);
      }

      SmartPtr<Vector> sol_p_c = Csol_x->GetCompNonConst(2);
      sol_p_c->Set(0.0);
      if (IsValid(sigma_tilde_p_c_inv)) {
        DBG_PRINT_VECTOR(2, "rhs_pc", *Crhs_x->GetComp(2));
        DBG_PRINT_VECTOR(2, "delta_y_c", sol_cR);
        DBG_PRINT_VECTOR(2, "Sig~_{p_c}^{-1}", *sigma_tilde_p_c_inv);
        sol_p_c->Copy(*Crhs_x->GetComp(2));
        sol_p_c->Axpy(1.0, sol_cR);
        sol_p_c->ElementWiseMultiply(*sigma_tilde_p_c_inv);
      }

      SmartPtr<Vector> sol_n_d = Csol_x->GetCompNonConst(3);
      sol_n_d->Set(0.0);
      if (IsValid(sigma_tilde_n_d_inv)) {
        pd_l->TransMultVector(-1.0, sol_dR, 0.0, *sol_n_d);
        sol_n_d->Axpy(1.0, *Crhs_x->GetComp(3));
        sol_n_d->ElementWiseMultiply(*sigma_tilde_n_d_inv);
      }

      SmartPtr<Vector> sol_p_d = Csol_x->GetCompNonConst(4);
      sol_p_d->Set(0.0);
      if (IsValid(sigma_tilde_p_d_inv)) {
        neg_pd_u->TransMultVector(-1.0, sol_dR, 0.0, *sol_p_d);
        sol_p_d->Axpy(1.0, *Crhs_x->GetComp(4));
        sol_p_d->ElementWiseMultiply(*sigma_tilde_p_d_inv);
      }
    }

    return status;

  }

  SmartPtr<const Vector>
  AugRestoSystemSolver::Neg_Omega_c_plus_D_c(
    const SmartPtr<const Vector>& sigma_tilde_n_c_inv,
    const SmartPtr<const Vector>& sigma_tilde_p_c_inv,
    const Vector* D_c,
    const Vector& any_vec_in_c)
  {
    DBG_START_METH("AugRestoSystemSolver::Neg_Omega_c_plus_D_c",dbg_verbosity);
    SmartPtr<Vector> retVec;
    if (IsValid(sigma_tilde_n_c_inv) || IsValid(sigma_tilde_p_c_inv) || D_c) {
      if (!neg_omega_c_plus_D_c_cache_.
          GetCachedResult3Dep(retVec, GetRawPtr(sigma_tilde_n_c_inv), GetRawPtr(sigma_tilde_p_c_inv), D_c)) {
        DBG_PRINT((2,"Not found in cache\n"));
        retVec = any_vec_in_c.MakeNew();
        retVec->Set(0.0);

        if (IsValid(sigma_tilde_n_c_inv)) {
          retVec->Axpy(1.0, *sigma_tilde_n_c_inv);
        }
        if (IsValid(sigma_tilde_p_c_inv)) {
          retVec->Axpy(1.0, *sigma_tilde_p_c_inv);
        }

        retVec->Scal(-1.0);

        if (D_c) {
          retVec->Axpy(1.0, *D_c);
        }

        neg_omega_c_plus_D_c_cache_.
        AddCachedResult3Dep(retVec, GetRawPtr(sigma_tilde_n_c_inv), GetRawPtr(sigma_tilde_p_c_inv), D_c);
      }
    }
    return ConstPtr(retVec);
  }

  SmartPtr<const Vector>
  AugRestoSystemSolver::Neg_Omega_d_plus_D_d(
    const Matrix& Pd_L,
    const SmartPtr<const Vector>& sigma_tilde_n_d_inv,
    const Matrix& neg_Pd_U,
    const SmartPtr<const Vector>& sigma_tilde_p_d_inv,
    const Vector* D_d,
    const Vector& any_vec_in_d)
  {
    DBG_START_METH("AugRestoSystemSolver::Neg_Omega_d_plus_D_d",dbg_verbosity);
    SmartPtr<Vector> retVec;
    if (IsValid(sigma_tilde_n_d_inv) || IsValid(sigma_tilde_p_d_inv) || D_d) {
      std::vector<const TaggedObject*> deps;
      std::vector<Number> scalar_deps;
      deps.push_back(&Pd_L);
      deps.push_back(GetRawPtr(sigma_tilde_n_d_inv));
      deps.push_back(&neg_Pd_U);
      deps.push_back(GetRawPtr(sigma_tilde_p_d_inv));
      deps.push_back(D_d);
      if (!neg_omega_d_plus_D_d_cache_.
          GetCachedResult(retVec, deps, scalar_deps)) {
        DBG_PRINT((2,"Not found in cache\n"));
        retVec = any_vec_in_d.MakeNew();
        retVec->Set(0.0);
        if (IsValid(sigma_tilde_n_d_inv)) {
          Pd_L.MultVector(1.0, *sigma_tilde_n_d_inv, 1.0, *retVec);
        }
        if (IsValid(sigma_tilde_p_d_inv)) {
          neg_Pd_U.MultVector(-1.0, *sigma_tilde_p_d_inv, 1.0, *retVec);
        }
        retVec->Scal(-1.0);
        if (D_d) {
          retVec->Copy(*D_d);
        }
        neg_omega_d_plus_D_d_cache_.
        AddCachedResult(retVec, deps, scalar_deps);
      }
    }
    return ConstPtr(retVec);
  }

  SmartPtr<const Vector> AugRestoSystemSolver::Sigma_tilde_n_c_inv(
    const SmartPtr<const Vector>& sigma_n_c,
    Number delta_x,
    const Vector& any_vec_in_c)
  {
    DBG_START_METH("AugRestoSystemSolver::Sigma_tilde_n_c_inv",dbg_verbosity);
    SmartPtr<Vector> retVec;
    if (IsValid(sigma_n_c) || delta_x != 0.0) {
      std::vector<const TaggedObject*> deps;
      std::vector<Number> scalar_deps;
      deps.push_back(GetRawPtr(sigma_n_c));
      scalar_deps.push_back(delta_x);
      if (!sigma_tilde_n_c_inv_cache_.GetCachedResult(retVec, deps, scalar_deps)) {
        DBG_PRINT((2,"Not found in cache\n"));
        retVec = any_vec_in_c.MakeNew();
        if (IsValid(sigma_n_c)) {
          retVec->Copy(*sigma_n_c);
          retVec->AddScalar(delta_x);
        }
        else {
          retVec->Set(delta_x);
        }
        retVec->ElementWiseReciprocal();

        sigma_tilde_n_c_inv_cache_.AddCachedResult(retVec, deps, scalar_deps);
      }
    }

    return ConstPtr(retVec);
  }

  SmartPtr<const Vector> AugRestoSystemSolver::Sigma_tilde_p_c_inv(
    const SmartPtr<const Vector>& sigma_p_c,
    Number delta_x,
    const Vector& any_vec_in_c)
  {
    DBG_START_METH("AugRestoSystemSolver::Sigma_tilde_p_c_inv",dbg_verbosity);
    SmartPtr<Vector> retVec;
    if (IsValid(sigma_p_c) || delta_x != 0.0) {
      std::vector<const TaggedObject*> deps;
      std::vector<Number> scalar_deps;
      deps.push_back(GetRawPtr(sigma_p_c));
      scalar_deps.push_back(delta_x);
      if (!sigma_tilde_p_c_inv_cache_.GetCachedResult(retVec, deps, scalar_deps)) {
        DBG_PRINT((2,"Not found in cache\n"));
        retVec = any_vec_in_c.MakeNew();
        if (IsValid(sigma_p_c)) {
          retVec->Copy(*sigma_p_c);
          retVec->AddScalar(delta_x);
        }
        else {
          retVec->Set(delta_x);
        }
        retVec->ElementWiseReciprocal();

        sigma_tilde_p_c_inv_cache_.AddCachedResult(retVec, deps, scalar_deps);
      }
    }

    return ConstPtr(retVec);
  }

  SmartPtr<const Vector> AugRestoSystemSolver::Sigma_tilde_n_d_inv(
    const SmartPtr<const Vector>& sigma_n_d,
    Number delta_x,
    const Vector& any_vec_in_n_d)
  {
    DBG_START_METH("AugRestoSystemSolver::Sigma_tilde_n_d_inv",dbg_verbosity);
    SmartPtr<Vector> retVec;
    if (IsValid(sigma_n_d) || delta_x != 0) {
      std::vector<const TaggedObject*> deps;
      std::vector<Number> scalar_deps;
      deps.push_back(GetRawPtr(sigma_n_d));
      scalar_deps.push_back(delta_x);
      if (!sigma_tilde_n_d_inv_cache_.GetCachedResult(retVec, deps, scalar_deps)) {
        DBG_PRINT((2,"Not found in cache\n"));
        retVec = any_vec_in_n_d.MakeNew();
        if (IsValid(sigma_n_d)) {
          retVec->Copy(*sigma_n_d);
          retVec->AddScalar(delta_x);
        }
        else {
          retVec->Set(delta_x);
        }
        retVec->ElementWiseReciprocal();

        sigma_tilde_n_d_inv_cache_.AddCachedResult(retVec, deps, scalar_deps);
      }
    }

    return ConstPtr(retVec);
  }

  SmartPtr<const Vector> AugRestoSystemSolver::Sigma_tilde_p_d_inv(
    const SmartPtr<const Vector>& sigma_p_d,
    Number delta_x,
    const Vector& any_vec_in_p_d)
  {
    DBG_START_METH("AugRestoSystemSolver::Sigma_tilde_p_d_inv",dbg_verbosity);
    SmartPtr<Vector> retVec;
    if (IsValid(sigma_p_d) || delta_x != 0) {
      std::vector<const TaggedObject*> deps;
      std::vector<Number> scalar_deps;
      deps.push_back(GetRawPtr(sigma_p_d));
      scalar_deps.push_back(delta_x);
      if (!sigma_tilde_p_d_inv_cache_.GetCachedResult(retVec, deps, scalar_deps)) {
        DBG_PRINT((2,"Not found in cache\n"));
        retVec = any_vec_in_p_d.MakeNew();

        if (IsValid(sigma_p_d)) {
          retVec->Copy(*sigma_p_d);
          retVec->AddScalar(delta_x);
        }
        else {
          retVec->Set(delta_x);
        }
        retVec->ElementWiseReciprocal();

        sigma_tilde_p_d_inv_cache_.AddCachedResult(retVec, deps, scalar_deps);
      }
    }

    return ConstPtr(retVec);
  }

  SmartPtr<const Vector> AugRestoSystemSolver::D_x_plus_wr_d(
    const SmartPtr<const Vector>& CD_x0,
    Number factor,
    const Vector& wr_d)
  {
    DBG_START_METH("AugRestoSystemSolver::D_x_plus_wr_d",dbg_verbosity);
    SmartPtr<Vector> retVec;

    std::vector<const TaggedObject*> deps;
    deps.push_back(&wr_d);
    if (IsValid(CD_x0)) {
      deps.push_back(GetRawPtr(CD_x0));
    }
    else {
      deps.push_back(&wr_d);
    }
    std::vector<Number> scalar_deps;
    scalar_deps.push_back(factor);

    if (!d_x_plus_wr_d_cache_.GetCachedResult(retVec, deps, scalar_deps)) {
      retVec = wr_d.MakeNew();
      retVec->Copy(wr_d);
      retVec->Scal(factor);
      if (IsValid(CD_x0)) {
        retVec->Axpy(1., *CD_x0);
      }
      d_x_plus_wr_d_cache_.AddCachedResult(retVec, deps, scalar_deps);
    }
    DBG_PRINT_VECTOR(2, "retVec", *retVec);
    return ConstPtr(retVec);
  }

  SmartPtr<const Vector> AugRestoSystemSolver::Rhs_cR(const Vector& rhs_c,
      const SmartPtr<const Vector>& sigma_tilde_n_c_inv, const Vector& rhs_n_c,
      const SmartPtr<const Vector>& sigma_tilde_p_c_inv, const Vector& rhs_p_c)
  {
    SmartPtr<Vector> retVec;
    std::vector<const TaggedObject*> deps;
    std::vector<Number> scalar_deps;
    deps.push_back(&rhs_c);
    deps.push_back(GetRawPtr(sigma_tilde_n_c_inv));
    deps.push_back(&rhs_n_c);
    deps.push_back(GetRawPtr(sigma_tilde_p_c_inv));
    deps.push_back(&rhs_p_c);
    if (!rhs_cR_cache_.GetCachedResult(retVec, deps, scalar_deps)) {
      retVec = rhs_c.MakeNew();
      retVec->Copy(rhs_c);

      SmartPtr<Vector> tmp = retVec->MakeNew();
      if (IsValid(sigma_tilde_n_c_inv)) {
        tmp->Copy(*sigma_tilde_n_c_inv);
        tmp->ElementWiseMultiply(rhs_n_c);
        retVec->Axpy(-1.0, *tmp);
      }

      if (IsValid(sigma_tilde_p_c_inv)) {
        tmp->Copy(*sigma_tilde_p_c_inv);
        tmp->ElementWiseMultiply(rhs_p_c);
        retVec->Axpy(1.0, *tmp);
      }
      rhs_cR_cache_.AddCachedResult(retVec, deps, scalar_deps);
    }
    return ConstPtr(retVec);
  }

  SmartPtr<const Vector> AugRestoSystemSolver::Rhs_dR(const Vector& rhs_d,
      const SmartPtr<const Vector>& sigma_tilde_n_d_inv, const Vector& rhs_n_d, const Matrix& pd_L,
      const SmartPtr<const Vector>& sigma_tilde_p_d_inv, const Vector& rhs_p_d, const Matrix& neg_pd_U)
  {
    SmartPtr<Vector> retVec;
    std::vector<const TaggedObject*> deps;
    std::vector<Number> scalar_deps;
    deps.push_back(&rhs_d);
    deps.push_back(GetRawPtr(sigma_tilde_n_d_inv));
    deps.push_back(&rhs_n_d);
    deps.push_back(&pd_L);
    deps.push_back(GetRawPtr(sigma_tilde_p_d_inv));
    deps.push_back(&rhs_p_d);
    deps.push_back(&neg_pd_U);
    if (!rhs_dR_cache_.GetCachedResult(retVec, deps, scalar_deps)) {
      retVec = rhs_d.MakeNew();
      retVec->Copy(rhs_d);

      if (IsValid(sigma_tilde_n_d_inv)) {
        SmartPtr<Vector> tmpn = sigma_tilde_n_d_inv->MakeNew();
        tmpn->Copy(*sigma_tilde_n_d_inv);
        tmpn->ElementWiseMultiply(rhs_n_d);
        pd_L.MultVector(-1.0, *tmpn, 1.0, *retVec);
      }

      if (IsValid(sigma_tilde_p_d_inv)) {
        SmartPtr<Vector> tmpp = sigma_tilde_p_d_inv->MakeNew();
        tmpp->Copy(*sigma_tilde_p_d_inv);
        tmpp->ElementWiseMultiply(rhs_p_d);
        neg_pd_U.MultVector(-1.0, *tmpp, 1.0, *retVec);
      }

      rhs_dR_cache_.AddCachedResult(retVec, deps, scalar_deps);
    }
    return ConstPtr(retVec);
  }

} // namespace Ipopt
