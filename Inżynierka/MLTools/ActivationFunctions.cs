using System;
using MathNet.Numerics.LinearAlgebra;

namespace MLTools
{
    public static class Derivatives
    {
        public static Vector<double> Linear(Vector<double> vector, double a)
        {
            for (int i = 0; i < vector.Count; i++)
            {
                vector[i] = a;
            }
            return vector;
        }
        public static Vector<double> Logistic(Vector<double> vector, double beta = 1f)
        {
            for (int i = 0; i < vector.Count; i++)
            {
                vector[i] = Functions.Logistic(vector[i],beta) * (1 - Functions.Logistic(vector[i],beta));
            }
            return vector;
        }
        public static Vector<double> TanH(Vector<double> vector, double beta = 1f)
        {
            for (int i = 0; i < vector.Count; i++)
            {
                vector[i] = 1 - Math.Pow(Math.Tanh(vector[i]), 2);
            }
            return vector;
        }
    }
    public static class Functions
    {
        public static Vector<double> Linear(Vector<double> vector, double a, double b = 0)
        {
            for (int i = 0; i < vector.Count; i++)
            {
                vector[i] = a * vector[i] + b;
            }
            return vector;
        }
        public static Vector<double> Logistic(Vector<double> vector, double beta = 1f)
        {
            for (int i = 0; i < vector.Count; i++)
            {
                vector[i] = Logistic(vector[i],beta);
            }
            return vector;
        }
        public static Vector<double> TanH(Vector<double> vector, double beta = 1f)
        {
            for (int i = 0; i < vector.Count; i++)
            {
                vector[i] = Math.Tanh(vector[i]);
            }
            return vector;
        }
        public static Vector<double> Binary(Vector<double> vector)
        {
            for (int i = 0; i < vector.Count; i++)
            {
                vector[i] = vector[i] >= 0 ? 1 : 0;
            }
            return vector;
        }
        public static double Logistic(double number, double beta)
        {
            return 1 / (1 + Math.Exp(-beta * number));
        }
    }
}
