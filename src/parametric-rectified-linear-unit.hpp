#ifndef IAD_2A_PARAMETRIC_RECTIFIED_LINEAR_UNIT_HPP
#define IAD_2A_PARAMETRIC_RECTIFIED_LINEAR_UNIT_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "activation-function.hpp"

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ///////////////////////////////////////////// | Class: RectifiedLinearUnit <
    class ParametricRectifiedLinearUnit
            : public ActivationFunction
    {
    public:
        //========================================================= | Methods <<
        //--------------------------------------------------- | Constructors <<<
        ParametricRectifiedLinearUnit
                () = delete;

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
    };
}

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_PARAMETRIC_RECTIFIED_LINEAR_UNIT_HPP
