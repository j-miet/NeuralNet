/*Tensor implementation
*/

namespace NeuralNet{

    /// <summary>
    /// Tensor class for 1D arrays
    /// </summary>
    class Tensor
    {
        private readonly double[] data;

        /// <summary>
        /// Initializes a new tensor object
        /// </summary>
        /// <param name="data">Data array of double type values</param>      
        public Tensor(double[] data)
        {
            this.data = data;
        }

        /// <summary>
        /// Returns data array
        /// </summary>
        /// <returns>Data array of all values</returns> 
        public double[] GetData()
        {
            return data;
        }

        /// <summary>
        /// Returns size of tensor
        /// </summary>
        /// <returns>Size of data array</returns>
        public int GetSize()
        {
            return data.Length;
        }

        /// <summary>
        /// Prints tensor size and all its values
        /// </summary>  
        public void PrintData()
        {
            int size = data.Length;
            Console.WriteLine("Size: " + size + ", ["+string.Join<double>(" ", data)+"]");
        }

        /// <summary>
        /// Sums tensor1 and tensor2 elementwise and returns this new tensor
        /// </summary>
        /// <param name="tensor1">Tensor object</param>
        /// <param name="tensor2">Tensor object</param>
        /// <returns>Tensor object</returns>
        /// <exception cref="Exception">Tensors have different size</exception>
        public static Tensor SumTensors(Tensor tensor1, Tensor tensor2)
        {
            if (tensor1.data.Length != tensor2.data.Length)
            {
                throw new Exception("Size of tensors don't match!");
            }
            else
            {
                double[] data1 = new double[tensor1.data.Length];
                Array.Copy(tensor1.data, data1, tensor1.data.Length);
                double[] data2 = new double[tensor2.data.Length];
                Array.Copy(tensor2.data, data2, tensor2.data.Length);

                for (int i = 0; i < data1.Length; i++)
                {
                    data1[i] += data2[i];
                }
                Tensor tensorObject = new(data1);
                return tensorObject;
            }
        }

        /// <summary>
        /// Multiplies all tensor values by -1 and returns this new tensor.
        /// </summary>
        /// <param name="tensor">Tensor object</param>
        /// <returns>Tensor object</returns>
        public static Tensor MinusTensor(Tensor tensor)
        {
            double[] data = new double[tensor.data.Length];
            Array.Copy(tensor.data, data, tensor.data.Length);
            for (int i = 0; i < tensor.data.Length; i++)
            {
               data[i] = -data[i];
            }
            Tensor tensorObject = new(data);
            return tensorObject;
        }

        /// <summary>
        /// Substracts tensor1 from tensor2 elementwise and return this new tensor
        /// </summary>
        /// <param name="tensor1">Tensor object</param>
        /// <param name="tensor2">Tensor object</param>
        /// <returns>Tensor object</returns>
        public static Tensor SubtractTensors(Tensor tensor1, Tensor tensor2)
        {
            return Tensor.SumTensors(tensor1, Tensor.MinusTensor(tensor2));
        }

        /// <summary>
        /// Returns a scaled tensor
        /// Each value is scaled multiplier, scalar*tensor[i]
        /// </summary>
        /// <param name="tensor">Tensor object</param>
        /// <param name="scalar">Scalar value</param>
        /// <returns>Tensor object</returns>   
        public static Tensor ScaleTensor(Tensor tensor, double scalar)
        {
            double[] data = new double[tensor.data.Length];
            Array.Copy(tensor.data, data, tensor.data.Length);

            for (int i = 0; i < data.Length; i++)
            {
                data[i] *= scalar;
            }

            Tensor tensorObject = new(data);
            return tensorObject;
        }

        /// <summary>
        /// Multiplies two tensors elementwise and returns this product tensor.
        /// </summary>
        /// <param name="tensor1">Tensor object</param>
        /// <param name="tensor2">Tensor object</param>
        /// <returns>Tensor object</returns>
        /// <exception cref="Exception">Tensors have different size</exception>
        public static Tensor MultiplyTensor(Tensor tensor1, Tensor tensor2)
        {
            if (tensor1.data.Length != tensor2.data.Length)
            {
                throw new Exception("Size of tensors don't match!");
            }
            else
            {
                double[] data1 = new double[tensor1.data.Length];
                Array.Copy(tensor1.data, data1, tensor1.data.Length);
                double[] data2 = new double[tensor2.data.Length];
                Array.Copy(tensor2.data, data2, tensor2.data.Length);

                for (int i = 0; i < data1.Length; i++)
                {
                    data1[i] *= data2[i];
                }
                Tensor tensorObject = new(data1);
                return tensorObject;
            }

        }

        /// <summary>
        /// Sums all values of a tensor
        /// </summary>
        /// <param name="tensor">Tensor Object</param>
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
        /// Calculates the dot product of two tensors
        /// </summary>
        /// <param name="tensor1">Tensor object</param>
        /// <param name="tensor2">Tensor object</param>
        /// <returns>Dot product</returns>
        /// <exception cref="Exception">Tensors have different size</exception>
        public static double DotProduct(Tensor tensor1, Tensor tensor2)
        {
            if (tensor1.data.Length != tensor2.data.Length)
            {
                throw new Exception("Size of tensors don't match!");
            }
            else
            {
                double dotProduct = 0; 
                for (int i = 0; i < tensor1.data.Length; i++)
                {
                    dotProduct += tensor1.data[i]*tensor2.data[i];
                }
                return dotProduct;
            }
        }
    }
}