using Random
using StatsBase
using Statistics
function partition_idxs(N,chunk_size)
    idxs = sample(1:N,N, replace = false)
    idxs_test = idxs[1:chunk_size]
    idxs_train = idxs[chunk_size+1:N]
    return idxs_test,idxs_train
end

function partition(data,labels,chunk_size)
    idxs = sort(sample(1:length(data),chunk_size, replace = false))
    data_mut = copy(data)
    labels_mut = copy(labels)
    data_test = data_mut[idxs]
    labels_test = labels_mut[idxs]
    data_train = deleteat!(data_mut,idxs)
    labels_train = deleteat!(labels_mut,idxs)
    return data_test,data_train,labels_test,labels_train
end
