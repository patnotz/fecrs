/*
 * typeinfo.hpp
 *
 *  Created on: Mar 24, 2011
 *      Author: pknotz
 */
#ifndef TYPEINFO_HPP_
#define TYPEINFO_HPP_

#include <Teuchos_ScalarTraits.hpp>
#include <Tpetra_DefaultPlatform.hpp>

//
// Specify types used in this example
//
typedef double Scalar;
typedef Teuchos::ScalarTraits<Scalar>::magnitudeType Magnitude;
typedef int Ordinal;
typedef Tpetra::DefaultPlatform::DefaultPlatformType Platform;
typedef Tpetra::DefaultPlatform::DefaultPlatformType::NodeType Node;
using Tpetra::global_size_t;

#endif /* TYPEINFO_HPP_ */
