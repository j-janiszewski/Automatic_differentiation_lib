using DrWatson
@quickactivate "ad_lib"

using MLDatasets: MNIST
using Printf
using Statistics
import Base: -
include(srcdir("operators.jl"))



function dense(w, b, x, activation)
    return activation(w * x .+ b)
end
function dense(w, x, activation)
    return activation(w * x)
end
function dense(w, x)
    return w * x
end

function sparse_categorical_crossentropy(y, actual_class)
    select(.-log.(y), actual_class)
end


-(x::Vector, y::Matrix) = vec(x .- y)

function update_var!(x::Variable, alpha)
    x.output = x.output - (x.gradient * alpha)
end

NUM_OF_CLASSES = 10
LEARNING_RATE = 0.001
EPOCHS = 5


# variables that will be modified
b = Variable(rand(Float64, NUM_OF_CLASSES), name="dense_layer_bias")
w = Variable(rand(Float64, (NUM_OF_CLASSES, 13 * 13)) ./ 10, name="dense_layer_weights")
w_conv = Variable(rand(Float32, 1, 9), name="convolution_weights")

train_dataset = MNIST(:train)
N = size(train_dataset.features)[3]

# Variables that will me modified on each run
img = Variable(train_dataset[1].features, name="img")
actual_class = Variable(train_dataset[1].targets + 1, name="actual_class")
# Layers
conv_layer = conv(img, w_conv, Constant(3), Constant(3), Constant(1))
max_pool_layer = maxpool(conv_layer, Constant(2))
flatten_layer = flatten(max_pool_layer)
fc_layer = dense(w, b, flatten_layer, softmax)
loss = sparse_categorical_crossentropy(fc_layer, actual_class)
net = topological_sort(loss)

# Training

@printf("Training network... \n")

losses = zeros(N)
for j = 1:EPOCHS
    for i = 1:N
        img.output = train_dataset[i].features
        actual_class.output = train_dataset[i].targets + 1
        loss_value = forward!(net)
        losses[i] = loss_value
        backward!(net)
        update_var!(b, LEARNING_RATE)
        update_var!(w, LEARNING_RATE)
        update_var!(w_conv, LEARNING_RATE)
    end
    @printf("Avarage loss during epoch #%d run : %f \n", j, mean(losses))
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

