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
#include <Tpetra_SerialPlatform.hpp>
#include <Tpetra_MpiPlatform.hpp>

#include <Kokkos_SerialNode.hpp>
#include <Kokkos_TBBNode.hpp>

using Tpetra::global_size_t;
typedef int OrdinalT;
typedef OrdinalT LocalOrdinalT;
typedef OrdinalT GlobalOrdinalT;
typedef double ScalarT;

typedef Teuchos::ScalarTraits<ScalarT>::magnitudeType MagnitudeT;
typedef Kokkos::TBBNode KNodeT;
//typedef Kokkos::SerialNode KNodeT;
typedef Tpetra::MpiPlatform<KNodeT> PlatformT;

typedef const Teuchos::Comm<int> CommT;
typedef Tpetra::Map<LocalOrdinalT, GlobalOrdinalT, KNodeT> MapT;
typedef Tpetra::CrsGraph<LocalOrdinalT, GlobalOrdinalT, KNodeT> GraphT;
typedef Tpetra::CrsMatrix<ScalarT, LocalOrdinalT, GlobalOrdinalT, KNodeT> MatrixT;

#endif /* TYPEINFO_HPP_ */
