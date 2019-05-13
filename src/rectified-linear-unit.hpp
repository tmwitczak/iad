#ifndef IAD_2A_RECTIFIED_LINEAR_UNIT_HPP
#define IAD_2A_RECTIFIED_LINEAR_UNIT_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "activation-function.hpp"

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ///////////////////////////////////////////// | Class: RectifiedLinearUnit <
    class RectifiedLinearUnit
            : public ActivationFunction
    {
    public:
        //========================================================= | Methods <<
        //--------------------------------------------------- | Constructors <<<
        RectifiedLinearUnit
                () = default;

        RectifiedLinearUnit
                (RectifiedLinearUnit const &) = default;

        RectifiedLinearUnit
                (RectifiedLinearUnit &&) = default;

        //------------------------------------------------------ | Operators <<<
        RectifiedLinearUnit &operator=
                (RectifiedLinearUnit const &) = default;

        RectifiedLinearUnit &operator=
                (RectifiedLinearUnit &&) = default;

        //----------------------------------------------------- | Destructor <<<
        ~RectifiedLinearUnit
                () noexcept override = default;

        //---------------------------- | Interface implementation: Cloneable <<<
        std::unique_ptr<ActivationFunction> clone
                () const override;

        //------------------- | Interface implementation: ActivationFunction <<<
        Eigen::ArrayXd operator()
                (Eigen::ArrayXd const &input) const override;

        Eigen::ArrayXd derivative
                (Eigen::ArrayXd const &input) const override;
    };
}

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_RECTIFIED_LINEAR_UNIT_HPP
