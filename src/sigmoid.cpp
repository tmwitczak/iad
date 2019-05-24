///////////////////////////////////////////////////////////////////// | Includes
#include "sigmoid.hpp"

/////////////////////////////////////////////////////////// | Using declarations
using Array = Eigen::ArrayXd;

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ///////////////////////////////////////////////////////// | Class: Sigmoid <
    //=========================================================== | Behaviour <<
    //------------------------------ | Interface: Cloneable | Implementation <<<
    std::unique_ptr<ActivationFunction> Sigmoid::clone
            () const
    {
        return std::make_unique<Sigmoid>(*this);
    }

    //--------------------- | Interface: ActivationFunction | Implementation <<<
    Array Sigmoid::operator()
            (Array const &input) const
    {
        return 1.0 / (1.0 + (-input).exp());
    }

    Array Sigmoid::derivative
            (Array const &input) const
    {
        Array sigmoidOutput = (*this)(input);
        return sigmoidOutput * (1.0 - sigmoidOutput);
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