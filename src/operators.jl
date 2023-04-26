import Base: *, sum, max, log
import LinearAlgebra: mul!
using LinearAlgebra

include(srcdir("graph_nodes.jl"))

*(A::GraphNode, x::GraphNode) = MatrixOperator(mul!, A, x)
forward(::MatrixOperator{typeof(mul!)}, A, x) = return A * x
backward(::MatrixOperator{typeof(mul!)}, A, x, g) = tuple(g * x', A' * g)

Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = MatrixOperator(*, x, y)
forward(::MatrixOperator{typeof(*)}, x, y) = return x .* y
backward(node::MatrixOperator{typeof(*)}, x, y, g) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(y .* 𝟏)
        Jy = diagm(x .* 𝟏)
        tuple(Jx' * g, Jy' * g)
    end


Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = MatrixOperator(-, x, y)
forward(::MatrixOperator{typeof(-)}, x, y) = return x .- y
backward(::MatrixOperator{typeof(-)}, x, y, g) = tuple(g, -g)

Base.Broadcast.broadcasted(-, x::GraphNode) = MatrixOperator(-, x)
forward(::MatrixOperator{typeof(-)}, x,) = return .-x
backward(::MatrixOperator{typeof(-)}, x, g) = tuple(-g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = MatrixOperator(+, x, y)
forward(::MatrixOperator{typeof(+)}, x, y) = return x .+ y
backward(::MatrixOperator{typeof(+)}, x, y, g) = tuple(g, g)


sum(x::GraphNode) = MatrixOperator(sum, x)
forward(::MatrixOperator{typeof(sum)}, x) = return sum(x)
backward(::MatrixOperator{typeof(sum)}, x, g) =
    let
        𝟏 = ones(length(x))
        J = 𝟏'
        tuple(J' * g)
    end

Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = MatrixOperator(/, x, y)
forward(::MatrixOperator{typeof(/)}, x, y) = return x ./ y
backward(node::MatrixOperator{typeof(/)}, x, y::Real, g) =
    let
        𝟏 = ones(length(node.output))
        Jx = diagm(𝟏 ./ y)
        Jy = (-x ./ y .^ 2)
        tuple(Jx' * g, Jy' * g)
    end


Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = MatrixOperator(max, x, y)
forward(::MatrixOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::MatrixOperator{typeof(max)}, x, y, g) =
    let
        Jx = diagm(isless.(y, x))
        Jy = diagm(isless.(x, y))
        tuple(Jx' * g, Jy' * g)
    end


Base.Broadcast.broadcasted(log, x::GraphNode) = MatrixOperator(log, x)
forward(::MatrixOperator{typeof(log)}, x) = return log.(x)
backward(::MatrixOperator{typeof(log)}, x, g) = tuple(((1 ./ x)' .* g)')


select(x::GraphNode, index) = MatrixOperator(select, x, index)
forward(::MatrixOperator{typeof(select)}, x, index) = return x[index]
backward(::MatrixOperator{typeof(select)}, x, index, g) =
    let
        result = zeros(size(x))
        result[index] = g
        tuple(result')
    end

softmax(x::GraphNode) = MatrixOperator(softmax, x)
forward(::MatrixOperator{typeof(softmax)}, x) = return exp.(x) ./ sum(exp.(x))
backward(node::MatrixOperator{typeof(softmax)}, x, g) =
    let
        y = node.output
        J = diagm(y) .- y * y'
        tuple(J' * g)
    end

flatten(x::GraphNode) = MatrixOperator(flatten, x)
forward(::MatrixOperator{typeof(flatten)}, x) = return vec(x)
backward(::MatrixOperator{typeof(flatten)}, x, g) =
    let
        M, N = size(x)
        tuple(reshape(g, M, N))
    end


function im2col(x::Matrix{Float32}, m::Int, n::Int, stride::Int)
    M, N = size(x)
    mc = (M - m) ÷ stride + 1
    nc = (N - n) ÷ stride + 1
    B = Array{Float32}(undef, m * n, mc * nc)
    for j = 1:nc
        for i = 1:mc
            @views block = x[((i-1)*stride+1):((i-1)*stride+1+m-1), (j-1)*stride+1:(j-1)*stride+1+n-1]
            for k = 1:m*n
                B[k, (j-1)*mc+i] = block[k]
            end
        end
    end
    B
end


conv(x::GraphNode, w::GraphNode, m::Constant, n::Constant, stride::Constant) = ConvOperator(conv, x, w, m, n, stride)
forward(conv_layer::ConvOperator{typeof(conv)}, x::Matrix{Float32}, w::Matrix{Float32}, m::Int, n::Int, stride::Int) =
    let
        M, N = size(x)
        b = im2col(x, m, n, stride)
        conv_layer.im2col = b
        reshape(w * b, (M - m) ÷ stride + 1, (N - n) ÷ stride + 1)
    end
backward(conv_layer::ConvOperator{typeof(conv)}, x::Matrix{Float32}, w::Matrix{Float64}, m::Int, n::Int, stride::Int, g) =
    let
        M, N = size(x)
        mc = (M - m) ÷ stride + 1
        nc = (N - n) ÷ stride + 1
        reshaped_grad = reshape(g, 1, mc * nc)
        dw = zeros(1, n * m)
        dx = zeros(size(x))
        for i = 1:mc*nc
            @views grad_block = reshaped_grad[1, i]
            @views im2col_block = conv_layer.im2col[:, i]
            dw += (im2col_block * grad_block)'
            row = grad_block * w
            x_pos = ((i - 1) ÷ mc) * stride + 1
            y_pos = ((i - 1) % nc) * stride + 1
            dx[x_pos:(x_pos+m-1), y_pos:(y_pos+n-1)] += reshape(row, m, n)
        end
        tuple(dx, dw)
    end



maxpool(x::GraphNode, n::Constant) = MatrixOperator(maxpool, x, n)
forward(::MatrixOperator{typeof(maxpool)}, x, n) =
    let
        M, N = size(x)
        M_out = 1 + (M - n) ÷ n
        N_out = 1 + (N - n) ÷ n
        out = zeros(M_out, N_out)
        for i = 1:n:M
            for j = 1:n:N
                out[(i+1)÷n, (j+1)÷n] = maximum(x[i:(i+n-1), j:(j+n-1)])
            end
        end
        out
    end
backward(::MatrixOperator{typeof(maxpool)}, x, n, g) =
    let
        M, N = size(x)
        M_out, N_out = size(g)
        dx = zeros(M, N)
        for i = 1:M_out
            for j = 1:N_out
                pool = x[1+(i-1)*n:i*n, 1+(j-1)*n:j*n]
                mask = (pool .== maximum(pool))
                dx[1+(i-1)*n:i*n, 1+(j-1)*n:j*n] = mask * g[i, j]
            end
        end
        tuple(dx)
    end