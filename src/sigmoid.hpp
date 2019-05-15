#ifndef IAD_2A_SIGMOID_HPP
#define IAD_2A_SIGMOID_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "activation-function.hpp"

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ///////////////////////////////////////////////////////// | Class: Sigmoid <
    class Sigmoid final
            : public ActivationFunction
    {
    public:
        // TODO: Fix serialization
        template <class Archive>
        void save(Archive &ar) const
        {}

        template <class Archive>
        void load(Archive &ar)
        {}

        //======================================================= | Behaviour <<
        //--------------------------------------------------- | Constructors <<<
        Sigmoid
                () = default;

        Sigmoid
                (Sigmoid const &) = default;

        Sigmoid
                (Sigmoid &&) = default;

        //------------------------------------------------------ | Operators <<<
        Sigmoid &operator=
                (Sigmoid const &) = default;

        Sigmoid &operator=
                (Sigmoid &&) = default;

        //----------------------------------------------------- | Destructor <<<
        ~Sigmoid
                () noexcept final = default;

        //-------------------------- | Interface: Cloneable | Implementation <<<
        std::unique_ptr<ActivationFunction> clone
                () const final;

        //----------------- | Interface: ActivationFunction | Implementation <<<
        Eigen::ArrayXd operator()
                (Eigen::ArrayXd const &input) const final;

        Eigen::ArrayXd derivative
                (Eigen::ArrayXd const &input) const final;
    };
}

CEREAL_REGISTER_TYPE(NeuralNetworks::Sigmoid)
CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::ActivationFunction,
                                     NeuralNetworks::Sigmoid)

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_SIGMOID_HPP
