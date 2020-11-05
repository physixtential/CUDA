// Define graph of work + dependencies

cudaGraphCreate(&graph);
cudaGraphAddNode(graph, kernel_a, {}, ...);
cudaGraphAddNode(graph, kernel_b, { kernel_a }, ...); // Waits for a
cudaGraphAddNode(graph, kernel_c, { kernel_a }, ...); // Waits for a
cudaGraphAddNode(graph, kernel_d, { kernel_b, kernel_c }, ...); // Waits for b and c

// Instantiate graph and apply optimizations

cudaGraphInstantiate(&instance, graph);

// Launch executable graph 100 times

for(int i=0; i<100; i++)
      cudaGraphLaunch(instance, stream);
