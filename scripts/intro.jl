using DrWatson
@quickactivate "ad_lib"


include(srcdir("operators.jl"))

weights = Variable(reshape(1:4, 1, 4), name="conv_weights")
img = Variable(reshape([0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0], 4, :), name="img")
order = topological_sort(conv(img, weights, Constant(2), Constant(2)))