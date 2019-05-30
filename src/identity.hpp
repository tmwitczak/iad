#ifndef IAD_2A_IDENTITY_HPP
#define IAD_2A_IDENTITY_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "activation-function.hpp"

#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    //////////////////////////////////////////////////////// | Class: Identity <
    class Identity final
            : public ActivationFunction
    {
    public:
        //======================================================= | Behaviour <<
        //--------------------------------------------------- | Constructors <<<
        Identity
                () = default;

        Identity
                (Identity const &) = default;

        Identity
                (Identity &&) = default;

        //------------------------------------------------------ | Operators <<<
        Identity &operator=
                (Identity const &) = default;

        Identity &operator=
                (Identity &&) = default;

        //----------------------------------------------------- | Destructor <<<
        ~Identity
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
CEREAL_REGISTER_TYPE(NeuralNetworks::Identity)
CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::ActivationFunction,
                                     NeuralNetworks::Identity)

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_IDENTITY_HPP
