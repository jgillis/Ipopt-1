// Copyright (C) 2004, International Business Machines and others.
// All Rights Reserved.
// This code is published under the Common Public License.
//
// $Id$
//
// Authors:  Carl Laird, Andreas Waechter     IBM    2004-08-13

#ifndef __IPCONVCHECK_HPP__
#define __IPCONVCHECK_HPP__

#include "IpUtils.hpp"
#include "IpAlgStrategy.hpp"

namespace Ipopt
{

  /** Base class for checking the algorithm
   *  termination criteria.
   */
  class ConvergenceCheck : public AlgorithmStrategyObject
  {
  public:
    /**@name Constructors/Destructors */
    //@{
    /** Constructor */
    ConvergenceCheck()
    {}

    /** Default destructor */
    virtual ~ConvergenceCheck()
    {}
    //@}

    /** Convergence return enum */
    enum ConvergenceStatus {
      CONTINUE,
      CONVERGED,
      MAXITER_EXCEEDED,
      FAILED
    };

    /** overloaded from AlgorithmStrategyObject */
    virtual bool InitializeImpl(const OptionsList& options,
                                const std::string& prefix) = 0;

    /** Pure virtual method for performing the convergence test */
    virtual ConvergenceStatus CheckConvergence()=0;

  private:
    /**@name Default Compiler Generated Methods
     * (Hidden to avoid implicit creation/calling).
     * These methods are not implemented and 
     * we do not want the compiler to implement
     * them for us, so we declare them private
     * and do not define them. This ensures that
     * they will not be implicitly created/called. */
    //@{
    /** Default Constructor */
    //    ConvergenceCheck();

    /** Copy Constructor */
    ConvergenceCheck(const ConvergenceCheck&);

    /** Overloaded Equals Operator */
    void operator=(const ConvergenceCheck&);
    //@}

  };

} // namespace Ipopt

#endif
