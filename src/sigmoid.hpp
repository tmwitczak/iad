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

    private:
        //======================================================= | Behaviour <<
        //------------------------------------------ | cereal: Serialization <<<
        friend class cereal::access;

        template <typename Archive>
        void save
                (Archive &archive) const
        {
        }

        template <typename Archive>
        void load
                (Archive &archive)
        {
        }
    };
}

//////////////////////////////////////// | cereal: Polymorphic type registration
CEREAL_REGISTER_TYPE(NeuralNetworks::Sigmoid)
CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::ActivationFunction,
                                     NeuralNetworks::Sigmoid)

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_SIGMOID_HPP
