

namespace NeuralNet{

    /// <summary>
    /// Tensor class for 2D arrays
    /// </summary>
    class Tensor
    {
        private readonly double[,] data;

        /// <summary>
        /// Initializes a new Tensor
        /// </summary>
        /// <param name="data">Data matrix of double type values</param>      
        public Tensor(double[,] data)
        {
            this.data = data;
        }

        /// <summary>
        /// Returns data matrix
        /// </summary>
        /// <returns>Data matrix of all values</returns> 
        public double[,] GetData()
        {
            return data;
        }

        /// <summary>
        /// Returns size of tensor as an array [rows, columns]
        /// </summary>
        /// <returns>Size of data matrix</returns>
        public int[] GetSize()
        {
            return [data.GetLength(0), data.GetLength(1)];
        }

        /// <summary>
        /// Prints tensor size and all its values
        /// </summary>  
        public void PrintData()
        {
            double[] rowValues = new double[data.GetLength(1)];
            Console.WriteLine("Size: ("+ GetSize()[0]+", "+GetSize()[1]+")");
            for (int row = 0; row < data.GetLength(0); row++)
            {
                for (int col = 0; col < data.GetLength(1); col++)
                {
                    rowValues[col] = data[row,col];
                }
                Console.WriteLine("Row " + row +": ["+string.Join<double>(" ", rowValues)+"]");
            }
        }

        /// <summary>
        /// Sums tensor1 and tensor2 elementwise and returns this new tensor
        /// </summary>
        /// <param name="tensor1">Tensor</param>
        /// <param name="tensor2">Tensor</param>
        /// <returns>Tensor</returns>
        /// <exception cref="Exception">Tensors have different size</exception>
        public static Tensor SumTensors(Tensor tensor1, Tensor tensor2)
        {
            int[] tensor1Size = tensor1.GetSize();
            int[] tensor2Size = tensor2.GetSize();
            
            if (tensor1Size[0] != tensor2Size[0] || tensor1Size[1] != tensor2Size[1])
            {
                throw new Exception("Size of tensors don't match!");
            }
            else
            {
                double[,] data1 = (double[,])tensor1.data.Clone();
                double[,] data2 = (double[,])tensor2.data.Clone();

                for (int row = 0; row < data1.GetLength(0); row++)
                {
                    for (int col = 0; col < data1.GetLength(1); col++)
                    {
                        data1[row,col] += data2[row,col];
                    }
                }
                Tensor tensorObject = new(data1);
                return tensorObject;
            }
        }

        /// <summary>
        /// Multiplies all tensor values by -1 and returns this new tensor.
        /// </summary>
        /// <param name="tensor">Tensor</param>
        /// <returns>Tensor</returns>
        public static Tensor MinusTensor(Tensor tensor)
        {
            double[,] data = (double[,])tensor.data.Clone();

            for (int row = 0; row < data.GetLength(0); row++)
            {
                for (int col = 0; col < data.GetLength(1); col++)
                {
                    data[row,col] = -data[row,col];
                }
            }
            Tensor tensorObject = new(data);
            return tensorObject;
        }

        /// <summary>
        /// Substracts tensor1 from tensor2 elementwise and return this new tensor
        /// </summary>
        /// <param name="tensor1">Tensor</param>
        /// <param name="tensor2">Tensor</param>
        /// <returns>Tensor</returns>
        public static Tensor SubtractTensors(Tensor tensor1, Tensor tensor2)
        {
            return SumTensors(tensor1, MinusTensor(tensor2));
        }

        /// <summary>
        /// Returns a scaled tensor
        /// Each value is scaled multiplier, scalar*tensor[i]
        /// </summary>
        /// <param name="tensor">Tensor</param>
        /// <param name="scalar">Scalar value</param>
        /// <returns>Tensor</returns>   
        public static Tensor ScaleTensor(Tensor tensor, double scalar)
        {
            double[,] data = (double[,])tensor.data.Clone();

            for (int row = 0; row < data.GetLength(0); row++)
            {
                for (int col = 0; col < data.GetLength(1); col++)
                {
                    data[row,col] *= scalar;
                }
            }
            Tensor tensorObject = new(data);
            return tensorObject;
        }

        /// <summary>
        /// Multiplies two tensors elementwise and returns this product tensor.
        /// </summary>
        /// <param name="tensor1">Tensor</param>
        /// <param name="tensor2">Tensor</param>
        /// <returns>Tensor</returns>
        /// <exception cref="Exception">Tensors have different size</exception>
        public static Tensor MultiplyTensor(Tensor tensor1, Tensor tensor2)
        {
            int[] tensor1Size = tensor1.GetSize();
            int[] tensor2Size = tensor2.GetSize();
            
            if (tensor1Size[0] != tensor2Size[0] || tensor1Size[1] != tensor2Size[1])
            {
                throw new Exception("Size of tensors don't match!");
            }
            else
            {
                double[,] data1 = (double[,])tensor1.data.Clone();
                double[,] data2 = (double[,])tensor2.data.Clone();

                    for (int row = 0; row < data1.GetLength(0); row++)
                {
                    for (int col = 0; col < data1.GetLength(1); col++)
                    {
                        data1[row,col] *= data2[row,col];
                    }
                }
                    Tensor tensorObject = new(data1);
                    return tensorObject;
            }

        }

        /// <summary>
        /// Sums all values of a tensor
        /// </summary>
        /// <param name="tensor">Tensor</param>
        /// <returns>Sum of values</returns>
        public static double SumValues(Tensor tensor)
        {
            double sumValue = 0;
            foreach (double value in tensor.data)
            {
                sumValue += value;   
            }
            return sumValue;
        }

        /// <summary>
        /// Calculates the product tensor
        /// </summary>
        /// <param name="tensor1">Tensor</param>
        /// <param name="tensor2">Tensor</param>
        /// <returns>Tensor</returns>
        /// <exception cref="Exception">tensor1 columns and tensor2 rows don't match in size</exception>
        public static Tensor TensorProduct(Tensor tensor1, Tensor tensor2)
        {
            int tensor1Cols = tensor1.GetSize()[1];
            int tensor2Rows = tensor2.GetSize()[0];
            int tensor1Rows = tensor1.GetSize()[0];
            int tensor2Cols = tensor2.GetSize()[1];

            if (tensor1Cols != tensor2Rows)
            {
                if (tensor1Cols == 1 || tensor1Rows == 1 && tensor2Rows == 1 && tensor2Cols == 1)
                {
                    // dot product
                    tensor2 = Transpose(tensor2);
                    tensor2Rows = tensor2.GetSize()[0];
                    tensor2Cols = tensor2.GetSize()[1];
                }
                else
                {
                    throw new Exception("tensor1 columns and tensor2 rows don't match");
                } 
            }
            double[,] data1 = (double[,])tensor1.data.Clone();
            double[,] data2 = (double[,])tensor2.data.Clone();

            double[,] productTensor = new double[tensor1Rows, tensor2Cols];
            for (int i = 0; i < tensor1Rows; i++)
            {
                for (int j = 0; j < tensor2Cols; j++)
                {
                    for (int k = 0; k < tensor2Rows; k++)
                    {
                        productTensor[i, j] += data1[i, k] * data2[k, j];
                    } 
                }
            }
            Tensor tensorObject = new(productTensor);
            return tensorObject;
        }

        /// <summary>
        /// Returns transposed tensor
        /// </summary>
        /// <param name="tensor">Tensor</param>
        /// <returns>Tensor</returns>
        public static Tensor Transpose(Tensor tensor)
        {
            int rowSize = tensor.GetSize()[0];
            int colSize = tensor.GetSize()[1];

            double[,] tensorArray = new double[colSize,rowSize];
            double[,] data = (double[,])tensor.data.Clone();

            for (int i = 0; i < rowSize; i++)
            {
                for (int j = 0; j < colSize; j++)
                {
                    tensorArray[j, i] = data[i ,j];
                }
            }
            Tensor tensorObject = new(tensorArray);
            return tensorObject;
        }

        /// <summary>
        /// Returns an array of rows from tensor, each as 1xn tensor object
        /// </summary>
        /// <param name="tensor">Tensor</param>
        /// <returns>Array of tensors</returns>
        public static Tensor[] TensorList(Tensor tensor)
        {
            double[,] tensorData = (double[,])tensor.GetData().Clone();
            Tensor[] tensorRows = new Tensor[tensor.GetSize()[0]];
            for (int row = 0; row < tensor.GetSize()[0]; row++)
            {
                double[,] tempRow = new double[tensor.GetSize()[1], 1];
                for (int col = 0; col < tensor.GetSize()[1]; col++)
                {
                    tempRow[col,0] = tensorData[row, col];
                }
                tensorRows[row] = new Tensor(tempRow);
            }
            return tensorRows;
        }
    }
}