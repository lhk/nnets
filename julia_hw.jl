# this is a first attempt at a Julia implementation of
# neural networks
# the code is almost equivalent to python.
# one central idea is to use dictionaries for the
# derivatives. look at the backward pass of the linear node to get an idea

println("starting")

arr=Array{Float64,2}

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
  return layer.W'*dy, derivs
end

type SoftMaxLoss
  #state
  x::arr
  probs::arr
  labels::Int # an array of indices, length is batch_size

  SoftMaxLoss() = new()
end

function forward!(layer::SoftMaxLoss, x::arr)
  layer.x=x
  exp_scores=exp(x)
  layer.probs=exp_scores/sum(exp_scores, 1)
  return -log(layer.probs)[layer.labels]
end

# attention, in most cases this will be the last node
# here the backward pass actually starts
# the value for d_y should be just ones
function backward(layer::SoftMaxLoss, dy::arr)
  dx=layer.probs
  dx[layer.labels]-=1
  return dx/10, Dict()
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



using MNIST
#using Gadfly

data=trainfeatures(12)
data=reshape(data, 28,28)


xs=[]
ys=[]

for x=1:28
  for y=1:28
    if data[x,y]>50
      push!(xs,x)
      push!(ys,y)
    end
  end
end

#creating the network

offset=1
scale=0.001

W1=abs(randn(200, 784)+offset)*scale
b1=abs(randn(200,1)+offset)*scale

W2=abs(randn(50, 200)+offset)*scale
b2=abs(randn(50,1)+offset)*scale

W3=abs(randn(10,50)+offset)*scale
b3=abs(randn(10,1)+offset)*scale

fc1=Lin(W1, b1)
fc2=Lin(W2, b2)
fc3=Lin(W3, b3)

softmax=SoftMaxLoss()


epochs=30
features=500

alpha=1e-4
lambda=1e-2

for e=1:epochs
  avg_loss=0
  avg_hits=0
  for f=1:features
    in_data=trainfeatures(f)
    in_data=reshape(in_data, 784, 1)
    in_label=round(Int, trainlabel(f))

    softmax.labels=in_label+1

    x1=forward!(fc1, in_data)
    x2=forward!(fc2, x1)
    x3=forward!(fc3, x2)

    predicted=indmax(x3)

    if predicted==in_label+1
      avg_hits+=1
    end


    out=forward!(softmax, x3)
    avg_loss+=out

    dy=ones(10,1)

    dict=Dict()
    dy, temp_dict=backward(softmax, dy)
    dict=merge(dict, temp_dict)

    dy, temp_dict=backward(fc3, dy)
    dict=merge(dict, temp_dict)

    dy, temp_dict=backward(fc2, dy)
    dict=merge(dict, temp_dict)

    dy, temp_dict=backward(fc1, dy)
    dict=merge(dict, temp_dict)

    fc1.W-=(dict[fc1.W] + norm(fc1.W)*lambda)*alpha
    fc1.b-=(dict[fc1.b] + norm(fc1.b)*lambda)*alpha
    fc2.W-=(dict[fc2.W] + norm(fc2.W)*lambda)*alpha
    fc2.b-=(dict[fc2.b] + norm(fc2.b)*lambda)*alpha
    fc3.W-=(dict[fc3.W] + norm(fc3.W)*lambda)*alpha
    fc3.b-=(dict[fc3.b] + norm(fc3.b)*lambda)*alpha

    #println(f)

  end
  avg_hits/=features
  avg_loss/=features
  println("average loss")
  println(avg_loss)
  println("average hits")
  println(avg_hits)
  println("--------------------")
end

println("finished")
