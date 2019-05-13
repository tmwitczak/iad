#ifndef IAD_2A_ACTIVATION_FUNCTION_HPP
#define IAD_2A_ACTIVATION_FUNCTION_HPP
///////////////////////////////////////////////////////////////////// | Includes
#include "cloneable.hpp"

#include <Eigen>
#include <memory>

//////////////////////////////////////////////////// | Namespace: NeuralNetworks
namespace NeuralNetworks
{
    //////////////////////////////////////////// | Interface: ActivationFunction
    class ActivationFunction
            : public Cloneable<ActivationFunction>
    {
    public:
        //========================================================== | Methods <
        //------------------------------------------------------ | Destructor <<
        ~ActivationFunction
                () noexcept override = 0;

        //----------------------------- | Interface implementation: Cloneable <<
        std::unique_ptr<ActivationFunction> clone
                () const override = 0;

        //-------------------------------------------------- | Main behaviour <<
        virtual Eigen::ArrayXd operator()
                (Eigen::ArrayXd const &input) const = 0;

        virtual Eigen::ArrayXd derivative
                (Eigen::ArrayXd const &input) const = 0;

    protected:
        //========================================================== | Methods <
        //---------------------------------------------------- | Constructors <<
        ActivationFunction
                () = default;

        ActivationFunction
                (ActivationFunction const &) = default;

        ActivationFunction
                (ActivationFunction &&) = default;

        //------------------------------------------------------- | Operators <<
        ActivationFunction &operator=
                (ActivationFunction const &) = default;

        ActivationFunction &operator=
                (ActivationFunction &&) = default;
    };
}

////////////////////////////////////////////////////////////////////////////////
#endif // IAD_2A_ACTIVATION_FUNCTION_HPP
