<h1>Diagrammatic Reasoning for Natural Language Processing</h1>
<h2>Introduction</h2>
<p>This code is an implementation of a natural language processing (NLP) task using Diagrammatic Reasoning. It uses the libraries Discopy, Sympy, Jax, and Lambeq to read, load, and parse sentences into diagrams, and then use these diagrams to make predictions using a circuit-based model.</p>
<h2>Implementation</h2>
<p>The code reads and loads the train and test data from the file 'mc_train_data.txt' and 'mc_test_data.txt' respectively, and it uses BobcatParser to parse the sentences into diagrams. Then, it creates an ansatz by assigning 2 dimensions to both noun and sentence spaces, and it converts the diagrams into circuits.
It then creates a function to calculate the loss and the gradient, and it uses the Jax library to optimize the parameters by minimizing the loss function. The optimizer runs for 90 epochs and the loss value is printed after each 10 epochs.</p>
<h2>Usage</h2>
<p>To use the code, you will need to have the libraries Discopy, Sympy, Jax, and Lambeq installed, and you will need to provide the missing files 'mc_train_data.txt' and 'mc_test_data.txt'.
You will also need to make sure that your environment is correctly set and that you have the necessary dependencies installed. Then, you can run the code in your Python environment.</p>
