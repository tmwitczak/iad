///////////////////////////////////////////////////////////////////// | Includes
#include "identity.hpp"

/////////////////////////////////////////////////////////// | Using declarations
using Array = Eigen::ArrayXd;

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    //////////////////////////////////////////////////////// | Class: Identity <
    //=========================================================== | Behaviour <<
    //------------------------------ | Interface: Cloneable | Implementation <<<
    std::unique_ptr<ActivationFunction> Identity::clone
            () const
    {
        return std::make_unique<Identity>(*this);
    }

    //--------------------- | Interface: ActivationFunction | Implementation <<<
    Array Identity::operator()
            (Array const &input) const
    {
        return input;
    }

    Array Identity::derivative
            (Array const &input) const
    {
        return Array::Ones(input.size());
    }

    //---------------------------------------------- | cereal: Serialization <<<
//    template <typename Archive>
//    void Sigmoid::save
//            (Archive &archive) const
//    {
//    }
//
//    template <typename Archive>
//    void Sigmoid::load
//            (Archive &archive)
//    {
//    }
}

////////////////////////////////////////////////////////////////////////////////