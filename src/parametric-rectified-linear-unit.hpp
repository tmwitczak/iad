#ifndef IAD_2A_PARAMETRIC_RECTIFIED_LINEAR_UNIT_HPP
#define IAD_2A_PARAMETRIC_RECTIFIED_LINEAR_UNIT_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "activation-function.hpp"

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ///////////////////////////////////////////// | Class: RectifiedLinearUnit <
    class ParametricRectifiedLinearUnit final
            : public ActivationFunction
    {
    public:
        //======================================================= | Behaviour <<
        //--------------------------------------------------- | Constructors <<<
        ParametricRectifiedLinearUnit
                ();

        explicit ParametricRectifiedLinearUnit
                (double const &parameter);

        ParametricRectifiedLinearUnit
                (ParametricRectifiedLinearUnit const &) = default;

        ParametricRectifiedLinearUnit
                (ParametricRectifiedLinearUnit &&) = default;

        //------------------------------------------------------ | Operators <<<
        ParametricRectifiedLinearUnit &operator=
                (ParametricRectifiedLinearUnit const &) = default;

        ParametricRectifiedLinearUnit &operator=
                (ParametricRectifiedLinearUnit &&) = default;

        //----------------------------------------------------- | Destructor <<<
        ~ParametricRectifiedLinearUnit
                () noexcept final = default;

        //---------------------------- | Interface implementation: Cloneable <<<
        std::unique_ptr<ActivationFunction> clone
                () const final;

        //------------------- | Interface implementation: ActivationFunction <<<
        Eigen::ArrayXd operator()
                (Eigen::ArrayXd const &input) const final;

        Eigen::ArrayXd derivative
                (Eigen::ArrayXd const &input) const final;

    private:
        //========================================================== | Fields <<
        double parameter;

        //======================================================= | Behaviour <<
        //------------------------------------------ | cereal: Serialization <<<
        friend class cereal::access;

        template <typename Archive>
        void save
                (Archive &archive) const
        {
            archive(parameter);
        }

        template <typename Archive>
        void load
                (Archive &archive)
        {
            archive(parameter);
        }
    };
}

//////////////////////////////////////// | cereal: Polymorphic type registration
CEREAL_REGISTER_TYPE(NeuralNetworks::ParametricRectifiedLinearUnit)
CEREAL_REGISTER_POLYMORPHIC_RELATION(NeuralNetworks::ActivationFunction,
        NeuralNetworks::ParametricRectifiedLinearUnit)

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_PARAMETRIC_RECTIFIED_LINEAR_UNIT_HPP
