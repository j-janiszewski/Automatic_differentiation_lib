using DrWatson
@quickactivate "ad_lib"

using MLDatasets: MNIST
using Printf
using Statistics
import Base: -
include(srcdir("operators.jl"))



function dense(w::Variable, b::Variable, x::Operator, activation::Function)
    return activation(w * x .+ b)
end
function dense(w::Variable, x::Operator, activation::Function)
    return activation(w * x)
end
function dense(w::Variable, x::Operator)
    return w * x
end

function sparse_categorical_crossentropy(y::Operator, actual_class::Variable)
    select(.-log.(y), actual_class)
end

function relu(x::GraphNode)
    max.(x, Constant(Float32(0.0)))
end

-(x::Vector, y::Matrix) = vec(x .- y)

# With RMSprop
function update_vars!(vars::Vector{Variable}, alpha::Float32, gamma::Float32, epsilon::Float32, epochs::Int)
    for i in eachindex(vars)
        #vars[i].output = vars[i].output - (vars[i].gradient * alpha)
        if isnothing(vars[i].rmsprop)
            vars[i].rmsprop = zeros(Float32, size(vars[i].gradient))
        end
        vars[i].rmsprop = gamma * vars[i].rmsprop .+ (1 - gamma) * vars[i].gradient .* vars[i].gradient
        vars[i].output = vars[i].output - vars[i].gradient .* (alpha ./ sqrt.(vars[i].rmsprop .+ epsilon))
    end
end

function train_and_test_mnist_cnn(learning_rate::Float32, gamma::Float32, epsilon::Float32, epochs::Int)
    NUM_OF_CLASSES = 10
    b = Variable(rand(Float32, NUM_OF_CLASSES), name="dense_layer_bias")
    w = Variable(rand(Float32, (NUM_OF_CLASSES, 13 * 13)) ./ 10, name="dense_layer_weights")
    w_conv = Variable(rand(Float32, 1, 9), name="convolution_weights")
    learnables = [b, w, w_conv]

    train_dataset = MNIST(:train)
    N = size(train_dataset.features)[3]

    img = Variable(train_dataset[1].features, name="img")
    actual_class = Variable(train_dataset[1].targets + 1, name="actual_class")
    #Layers
    conv_layer = relu(conv(img, w_conv, Constant(3), Constant(3), Constant(1)))
    max_pool_layer = maxpool(conv_layer, Constant(2))
    flatten_layer = flatten(max_pool_layer)
    fc_layer = dense(w, b, flatten_layer, softmax)
    loss = sparse_categorical_crossentropy(fc_layer, actual_class)
    net = topological_sort(loss)



    @printf("Training network... \n")

    losses = zeros(N)
    for j = 1:epochs
        for i = 1:N
            img.output = train_dataset[i].features
            actual_class.output = train_dataset[i].targets + 1
            loss_value = forward!(net)
            losses[i] = loss_value
            backward!(net)
            update_vars!(learnables, learning_rate, gamma, epsilon, j)
        end
        @printf("Avarage loss during epoch #%d run : %f \n", j, mean(losses))
        for i in eachindex(learnables)
            learnables[i].rmsprop = nothing
        end
    end

    test_dataset = MNIST(:test)
    N = size(test_dataset.features)[3]
    net = topological_sort(fc_layer)

    @printf("Testing network...\n")
    let count = 0
        for i = 1:N
            img.output = test_dataset[i].features
            y = forward!(net)
            if argmax(y) == (test_dataset[i].targets + 1)
                count += 1
            end
        end

        @printf("Network accuracy:  %f \n", count / N)
    end

end



const LEARNING_RATE = Float32(0.001)
const GAMMA = Float32(0.9)
const EPSILON = Float32(1e-6)
const EPOCHS = 3

@time train_and_test_mnist_cnn(LEARNING_RATE, GAMMA, EPSILON, EPOCHS)

