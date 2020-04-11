using Flux,Zygote
using OpenAIGym
import Reinforce.action
import Flux.params
using LinearAlgebra
using Flux.Optimise: update!
using Distributions
using Zygote:@adjoint

mutable struct PendulumPolicy <: Reinforce.AbstractPolicy
   train::Bool
   function PendulumPolicy(train=true)
     new(train)
   end
end

env = GymEnv("Pendulum-v0")

state_size = length(env.state)
action_size = length(env.actions)
action_bound = Float32(env.actions.hi[1])
batch_size = 12
mem_size = 10000
gamma = 0.9
rho = 0.1
alpha_act = 0.001
alpha_crit = 0.01
max_ep = 100
max_ep_len = 100
max_frames = 120
memory = []
frames = 0
eps = randn(action_size,batch_size)

println(action_size)
println(state_size)
actor_b = Chain(Dense(state_size, 25, relu),Dense(25,action_size),softmax)
actor_w = Chain(Dense(state_size, 25, relu),Dense(25,batch_size),softmax)
actor = Chain(x->(actor_b(x) .+ eps*actor_w(x)),x->tanh.(x),x->x*action_bound)
critic1 = Chain(Dense(state_size+action_size, 25, relu),Dense(25,25,relu),Dense(25,1))
critic2 = Chain(Dense(state_size+action_size, 25, relu),Dense(25,25,relu),Dense(25,1))

critic1_tgt = deepcopy(critic1)
critic2_tgt = deepcopy(critic2)

function update_target!(target,model; rho = 0.01)
   for p = 1:3
       target[p].b .= rho*model[p].b .+ (1 - rho)*target[p].b    # updation of target Actor Network
       target[p].W .= rho*model[p].W .+ (1 - rho)*target[p].W
   end 
end
function Log_likelihood(actor_w,actor_b,eps,alpha,x)
   llike1 = []
   g = actor_b(x) .+ actor_w(x).*eps
   for i=1:batch_size
     h = loglikelihood(Normal(), eps[:,i])+log(abs(det(actor_w(x))))
     for j=1:action_size
       h += log(1+(tanh(g[j,i]))^2)
     end
     push!(llike1,-h)
   end
   llike1 .= alpha*llike1
   llike = reshape(llike1,1,batch_size)
   return llike
end

@adjoint Log_likelihood(actor_w,actor_b,eps,alpha,x)=Log_likelihood_grad(actor_w,actor_b,eps,alpha,x), Δ -> Log_likelihood_grad(actor_w,actor_b,eps,alpha,x)' * Δ

function train()
   minibatch = sample(memory,batch_size)
   x = hcat(minibatch...)
   s = hcat(x[1,:]...)
   a = hcat(x[2,:]...)
   r = hcat(x[3,:]...)
   s1 = hcat(x[4,:]...)
   d = .!hcat(x[5,:]...)
   
   a1 = actor(s1)
   x1 = vcat(s1,a1)
   q1 = critic1_tgt(x1)
   q2 = critic2_tgt(x1)
   y1 = r + gamma*((sum(q1)<sum(q2) ? q1 : q2)-Log_likelihood(actor_w,actor_b,eps,0.01,s1)).*d
   x2 = vcat(s,a)
   v1 = critic1(x2)
   v2 = critic2(x2)
   loss_critic1 = (Flux.mse(v1,y1))/batch_size
   loss_critic2 = (Flux.mse(v2,y1))/batch_size
   opt = Descent(0.1)
   grads1 = gradient(()->loss_critic1,Flux.params(critic1))
   grads2 = gradient(()->loss_critic2,Flux.params(critic2))
   update!(opt, Flux.params(critic1) , grads1)
   update!(opt, Flux.params(critic2) , grads2)
    
   a2 = actor(s)
   x3 = vcat(s,a2)
   v3 = critic1(x3)
   v4 = critic2(x3)

   loss_policy = (sum((sum(v3)<sum(v4) ? v3 : v4)-Log_likelihood(actor_w,actor_b,eps,0.01,s)))/batch_size
   opt1 = Descent(-0.1)
   grads3 = gradient(()->loss_policy,Flux.params(actor,actor_w,actor_b))
   update!(opt1, Flux.params(actor,actor_w,actor_b) , grads3)
end
function replay_buffer(state, action, reward, next_state, done)
    if length(memory)>= mem_size
        deleteat!(memory,1)
    end
    push!(memory,[state, action, reward, next_state, done])
end
function action(pie::PendulumPolicy, reward, state, action)
    state = reshape(state, size(state)...,1)
    act_pred = actor(state)
    clamp.(act_pred[:,1],-action_bound,action_bound)
end
function episode!(env, pie=RandomPolicy())
    global frames
    ep = Episode(env, pie)
    frm = 0
    for (s,a,r,s1) in ep
      #OpenAIGym.render(env)
      r = env.done ? -1 : r
      if pie.train 
        replay_buffer(s,a,r,s1,env.done)
      end
      frames+=1
      frm+=1

      if length(memory) >= batch_size && pie.train
         train()
         update_target!(critic1_tgt, critic1;rho=rho)
         update_target!(critic2_tgt, critic2;rho=rho)
      end
    end
     ep.total_reward
end

sc = zeros(100)
e1 = 1
while e1 <= max_ep
   e1 = 1
   idx = 1
   reset!(env)
   total_reward = episode!(env, PendulumPolicy())
   sc[idx] = total_reward
   idx = idx%100 +1
   avg = mean(sc)
   println("Episode: $e1| Score: $total_reward|Avg score: $avg")
   e1+=1
end













