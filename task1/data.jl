using JLD2, FileIO

data1() = load("data.jld2", "u0s1", "rds1", "rdns1", "yreals1")
data2() = load("data.jld2", "u0s2", "rds2", "rdns2", "yreals2")
data3() = load("data.jld2", "u0s3", "rds3", "rdns3", "yreals3")
data4() = load("data.jld2", "u0s4", "rds4", "rdns4", "yreals4")

data_test1() = load("data.jld2", "u0s1test", "rds1test", "rdns1test")
data_test2() = load("data.jld2", "u0s2test", "rds2test", "rdns2test")
data_test3() = load("data.jld2", "u0s3test", "rds3test", "rdns3test")
data_test4() = load("data.jld2", "u0s3test", "rds3test", "rdns3test")

# Save the output of the model in result(i) files
save_trajectories(ys,i) = save("result$i.jld2", "result", ys)
