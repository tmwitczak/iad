#ifndef IAD_2A_ACTIVATION_FUNCTION_HPP
#define IAD_2A_ACTIVATION_FUNCTION_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "cloneable.hpp"

#include <Eigen>
#include <memory>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    ////////////////////////////////////////// | Interface: ActivationFunction <
    class ActivationFunction
            : private Cloneable<ActivationFunction>
    {
    public:
        //======================================================= | Behaviour <<
        //----------------------------------------------------- | Destructor <<<
        ~ActivationFunction
                () noexcept override = 0;

        //-------------------------- | Interface: Cloneable | Implementation <<<
        std::unique_ptr<ActivationFunction> clone
                () const override = 0;

        //----------------------------------------------------------- | Main <<<
        virtual Eigen::ArrayXd operator()
                (Eigen::ArrayXd const &input) const = 0;

        virtual Eigen::ArrayXd derivative
                (Eigen::ArrayXd const &input) const = 0;

    protected:
        //======================================================= | Behaviour <<
        //--------------------------------------------------- | Constructors <<<
        ActivationFunction
                () = default;

        ActivationFunction
                (ActivationFunction const &) = default;

        ActivationFunction
                (ActivationFunction &&) = default;

        //------------------------------------------------------ | Operators <<<
        ActivationFunction &operator=
                (ActivationFunction const &) = default;

        ActivationFunction &operator=
                (ActivationFunction &&) = default;
    };
}

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_ACTIVATION_FUNCTION_HPP
