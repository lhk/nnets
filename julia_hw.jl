# this is a first attempt at a Julia implementation of
# neural networks
# the code is almost equivalent to python.
# one central idea is to use dictionaries for the
# derivatives. look at the backward pass of the linear node to get an idea

arr=Array{Float16,2}

type Lin
  #parameters
  W::arr
  b::arr

  #param derivatives
  dW::arr
  db::arr

  #state
  x::arr

  Lin(W,b) = new(W,b)
end

#forward pass for linear layer
function forward!(layer::Lin, x::arr)
  layer.x=x
  return layer.W*x+layer.b
end

#attention this depends on the state
function backward(layer::Lin, dy::arr)
  derivs=Dict(
  layer.W => dy * layer.x',
  layer.b => dy,
  )
  return W'*y, derivs
end

type SoftMaxLoss

  #state
  x::arr
  probs::arr
  labels::Array{Int64,1} # an array of indices, length is batch_size

  SoftMaxLoss() = new()
end

function forward(layer::SoftMaxLoss, x::arr)
  layer.x=x
  exp_scores=exp(x)
  layer.probs=exp_scores/sum(exp_scores, 1)
  return -log(layer.probs[layer.labels, collect(1:1:size(labels)[1])])
end

# attention, in most cases this will be the last node
# here the backward pass actually starts
# the value for d_y should be just ones
function backward(layer::SoftMaxLoss, d_y)
  dscores=layer.probs*d_y
  dscores[layer.labels, collect(1:1:size(layer.labels)[1])]-=14
  dscores/=size(layer.x)[2]
  return dscores, Dict()
end

#computes a forward pass of a list of layers
#attention, this will change the state of the layers
function forward_pass(layers, x)
  for layer in layers
    x=forward(layer, x)
  end
  return x
end

#computes a backward pass
function backward_pass(layers, dy)
dparams=Dict()
  for layer in layers
  end
end

function t()
  1,2
end
println(2)
