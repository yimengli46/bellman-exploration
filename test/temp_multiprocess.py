from multiprocessing import Pool, current_process, Queue



def foo(filename):
    gpu_id = queue.get()
    try:
        # run processing on GPU <gpu_id>
        ident = current_process().ident
        print('{}: starting process on GPU {} for {}'.format(ident, gpu_id, filename))
        # ... process filename
        print('{}: finished'.format(ident))
    finally:
        queue.put(gpu_id)



def main():


    # initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    pool = Pool(processes=PROC_PER_GPU * NUM_GPUS)
    files = ['file{}.xyz'.format(x) for x in range(18)]
    for _ in pool.map(foo, files):
        pass
    pool.close()
    pool.join()


if __name__ == "__main__":
    NUM_GPUS = 4
    PROC_PER_GPU = 1  

    queue = Queue()

    main()