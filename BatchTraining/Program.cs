using System;
using System.Collections.Generic;
using System.Linq;
using CNTK;
using System.IO;

namespace BatchTraining
{
    class BatchTraining
    {
        const int hiddenNeuronCount = 3;
        const int batchSize = 50;
        const int epochCount = 20000;

        readonly Variable x;
        readonly Function y;

        public BatchTraining()
        {
            // Build graph
            x = Variable.InputVariable(new int[] { DataSet.InputSize, 1 }, DataType.Float);
            Parameter w1 = new Parameter(new int[] { hiddenNeuronCount, DataSet.InputSize }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter b = new Parameter(new int[] { hiddenNeuronCount, 1 }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            Parameter w2 = new Parameter(new int[] { DataSet.OutputSize, hiddenNeuronCount }, DataType.Float, CNTKLib.GlorotNormalInitializer());
            y = CNTKLib.Sigmoid(CNTKLib.Times(w2, CNTKLib.Sigmoid(CNTKLib.Plus(CNTKLib.Times(w1, x), b))));
        }

        public void Train(DataSet ds)
        {
            // Extend graph
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);
            Function loss = CNTKLib.BinaryCrossEntropy(y, yt);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);

            Learner learner = CNTKLib.SGDLearner(new ParameterVector(y.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, batchSize));
            Trainer trainer = Trainer.CreateTrainer(y, loss, y_yt_equal, new List<Learner>() { learner });

            // Train
            for (int epochI = 0; epochI <= epochCount; epochI++)
            {
                double sumLoss = 0;
                double sumEval = 0;

                //ds.Shuffle();
                for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
                {
                    Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize), DeviceDescriptor.CPUDevice);
                    Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);
                    var inputDataMap = new Dictionary<Variable, Value>()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                    trainer.TrainMinibatch(inputDataMap, false, DeviceDescriptor.CPUDevice);
                    sumLoss += trainer.PreviousMinibatchLossAverage() * trainer.PreviousMinibatchSampleCount();
                    sumEval += trainer.PreviousMinibatchEvaluationAverage() * trainer.PreviousMinibatchSampleCount();
                }
                Console.WriteLine(String.Format("{0}\tloss:{1}\teval:{2}", epochI, sumLoss / ds.Count, sumEval / ds.Count));
            }
        }

        public double Evaluate(DataSet ds)
        {
            // Extend graph
            Variable yt = Variable.InputVariable(new int[] { DataSet.OutputSize }, DataType.Float);

            Function y_rounded = CNTKLib.Round(y);
            Function y_yt_equal = CNTKLib.Equal(y_rounded, yt);
            Evaluator evaluator = CNTKLib.CreateEvaluator(y_yt_equal);

            double sumEval = 0;
            for (int batchI = 0; batchI < ds.Count / batchSize; batchI++)
            {
                Value x_value = Value.CreateBatch(x.Shape, ds.Input.GetRange(batchI * batchSize * DataSet.InputSize, batchSize * DataSet.InputSize), DeviceDescriptor.CPUDevice);
                Value yt_value = Value.CreateBatch(yt.Shape, ds.Output.GetRange(batchI * batchSize * DataSet.OutputSize, batchSize * DataSet.OutputSize), DeviceDescriptor.CPUDevice);
                var inputDataMap = new UnorderedMapVariableValuePtr()
                    {
                        { x, x_value },
                        { yt, yt_value }
                    };

                sumEval += evaluator.TestMinibatch(inputDataMap, DeviceDescriptor.CPUDevice) * batchSize;
            }
            return sumEval / ds.Count;
        }
    }

    public class DataSet
    {
        public const int InputSize = 16;
        public List<float> Input { get; set; } = new List<float>();

        public const int OutputSize = 1;
        public List<float> Output { get; set; } = new List<float>();

        public int Count { get; set; }

        public DataSet(string filename)
        {
            LoadData(filename);
        }

        void LoadData(string filename)
        {
            Count = 0;
            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = Normalize(line.Split('\t').Select(x => float.Parse(x)).ToList());
                Input.AddRange(floats.GetRange(1, InputSize));
                Output.Add(floats[0]);
                Count++;
            }
        }

        public void Shuffle()
        {
            Random rnd = new Random();
            for (int swapI = 0; swapI < Count; swapI++)
            {
                var a = rnd.Next(Count);
                var b = rnd.Next(Count);
                if (a != b)
                {
                    float T;
                    for (int i = 0; i < InputSize; i++)
                    {
                        T = Input[a * InputSize + i];
                        Input[a * InputSize + i] = Input[b * InputSize + i];
                        Input[b * InputSize + i] = T;
                    }
                    T = Output[a]; Output[a] = Output[b]; Output[b] = T;
                }
            }
        }

        static float[] minValues;
        static float[] maxValues;

        public static List<float> Normalize(List<float> floats)
        {
            List<float> normalized = new List<float>();
            for (int i = 0; i < floats.Count; i++)
                normalized.Add((floats[i] - minValues[i]) / (maxValues[i] - minValues[i]));
            return normalized;
        }

        public static void LoadMinMax(string filename)
        {
            foreach (String line in File.ReadAllLines(filename))
            {
                var floats = line.Split('\t').Select(x => float.Parse(x)).ToList();
                if (minValues == null)
                {
                    minValues = floats.ToArray();
                    maxValues = floats.ToArray();
                }
                else
                {
                    for (int i = 0; i < floats.Count; i++)
                        if (floats[i] < minValues[i])
                            minValues[i] = floats[i];
                        else
                            if (floats[i] > maxValues[i])
                            maxValues[i] = floats[i];
                }
            }
        }
    }

    public class Program
    {
        static void Main(string[] args)
        {
            DataSet.LoadMinMax(@"draft.txt");
            DataSet trainDS = new DataSet(@"draft.txt");
            DataSet testDS = new DataSet(@"draft.txt");

            BatchTraining app = new BatchTraining();
            app.Train(trainDS);
            Console.WriteLine("Eval train:" + app.Evaluate(trainDS));
            Console.WriteLine("Eval test:" + app.Evaluate(testDS));
        }
    }
}
