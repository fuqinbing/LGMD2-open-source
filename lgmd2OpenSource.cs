/*
 * Filename: lgmd2OpenSource
 * Author: Dr Qinbing FU
 * Date: 2019
 * Affiliation: qifu@lincoln.ac.uk
 * Organisation: University of Lincoln, Lincoln LN6 7TS, United Kingdom
 */



using System;


namespace LGMD
{
    /// <summary>
    /// This class describes the implementation of the LGMD2 based visual neural network corresponding to the following paper:
    /// [1] Q. Fu, C. Hu, J. Peng, F. C. Rind, S. Yue, "A Robust Collision Perception Visual Neural Network with Specific Selectivity to Darker Objects," 2019.
    /// Compared to previous related works, we highlight the modelling of biased-ON and OFF pathways and an adaptive inhibition mechanism.
    /// This class only presents the parameters setting and algorithms of this neural vision system, which can used in other applications.
    /// </summary>
    internal class lgmd2OpenSource
    {
        #region FIELD
        /// <summary>
        /// width of input image frame
        /// </summary>
        protected readonly int width;
        /// <summary>
        /// height of input image frame
        /// </summary>
        protected readonly int height;
        /// <summary>
        /// number of processing local cells (or pixels of the input image frame)
        /// </summary>
        protected readonly int Ncell;
        /// <summary>
        /// radius in convolving matrix, normally set to 1
        /// </summary>
        protected readonly int Np;
        /// <summary>
        /// number of spikes in a specified short time window
        /// </summary>
        protected int Nsp;
        /// <summary>
        /// number of successive  time steps constituting a short time window to calculate spiking frequency
        /// </summary>
        protected readonly int Nts;
        /// <summary>
        /// time constant in high-pass filtering of spike frequency adaptation mechanism
        /// </summary>
        protected readonly int tau_sfa;
        /// <summary>
        /// delay in high-pass of spike frequency adaptation mechanism
        /// </summary>
        protected readonly float hp_sfa;
        /// <summary>
        /// threshold in photorecptor mediation mechanism
        /// </summary>
        protected readonly int Tpm;
        /// <summary>
        /// scale parameter in spiking mechanism
        /// </summary>
        protected readonly int Cspi;
        /// <summary>
        /// clip point in ON and OFF rectifiers, normally set to a small real number or 0
        /// </summary>
        protected readonly float clip_point;
        /// <summary>
        /// weighting on local ON excitation
        /// </summary>
        protected readonly float W_on;
        /// <summary>
        /// weighting on local OFF excitation
        /// </summary>
        protected readonly float W_off;
        /// <summary>
        ///weighting on interaction of local ON/OFF excitation
        /// </summary>
        protected readonly float W_onoff;
        /// <summary>
        /// spiking threshold
        /// </summary>
        protected readonly float Tsp;
        /// <summary>
        /// threshold in spike frequency adaptation mechanism
        /// </summary>
        protected readonly float Tsfa;
        /// <summary>
        /// local bias on lateral ON inhibitions
        /// </summary>
        protected float W_i_on;
        /// <summary>
        /// baseline of local bias on lateral ON inhibitions
        /// </summary>
        protected float W_i_on_base;
        /// <summary>
        /// local bias on lateral OFF inhibitions
        /// </summary>
        protected float W_i_off;
        /// <summary>
        /// baseline of local bias on lateral OFF inhibitions
        /// </summary>
        protected float W_i_off_base;
        /// <summary>
        /// proportion in ON and OFF mechanisms
        /// </summary>
        protected readonly float dc;
        /// <summary>
        /// coefficient in sigmoid transformation
        /// </summary>
        protected readonly float Csig;
        /// <summary>
        /// convolution matrix in ON pathway
        /// </summary>
        protected float[,] Conv_ON;
        /// <summary>
        /// convolution matrix in OFF pathway
        /// </summary>
        protected float[,] Conv_OFF;
        /// <summary>
        /// convolution matrix in Grouping layer
        /// </summary>
        protected float[,] W_g;
        /// <summary>
        /// lateral inhibitions in ON pathway
        /// </summary>
        protected float[,] Inh_ON;
        /// <summary>
        /// lateral inhibitions in OFF pathway
        /// </summary>
        protected float[,] Inh_OFF;
        /// <summary>
        /// time constant in ON pathway
        /// </summary>
        protected readonly float[] tau_ON;
        /// <summary>
        /// time constant in OFF pathway
        /// </summary>
        protected readonly float[] tau_OFF;
        /// <summary>
        /// time constant in PM pathway
        /// </summary>
        protected readonly float tau_PM;
        /// <summary>
        /// delay in low-pass filtering of ON channels
        /// </summary>
        protected readonly float[] lp_ON;
        /// <summary>
        /// delay in low-pass filtering of OFF channels
        /// </summary>
        protected readonly float[] lp_OFF;
        /// <summary>
        /// delay in low-pass filtering of PM pathway
        /// </summary>
        protected float lp_PM;
        /// <summary>
        /// constant to compute the scale in Grouping layer
        /// </summary>
        protected readonly int Cw;
        /// <summary>
        /// small real number to compute the scale in Grouping layer
        /// </summary>
        protected readonly float Delta_C;
        /// <summary>
        /// decay coefficient
        /// </summary>
        protected readonly float Cde;
        /// <summary>
        /// decay threshold
        /// </summary>
        protected readonly int Tde;
        /// <summary>
        /// constant time interval in image stream
        /// </summary>
        protected float time_interval;
        /// <summary>
        /// photoreceptors layer
        /// </summary>
        protected int[,,] photoreceptors;
        /// <summary>
        /// ON cells
        /// </summary>
        protected float[,,] ons;
        /// <summary>
        /// OFF cells
        /// </summary>
        protected float[,,] offs;
        /// <summary>
        /// local ON summation cell
        /// </summary>
        protected float S_on; //local ON summation cell
        /// <summary>
        /// local OFF summation cell
        /// </summary>
        protected float S_off;
        /// <summary>
        /// Summation layer
        /// </summary>
        protected float[,] scells;
        /// <summary>
        /// Grouping layer
        /// </summary>
        protected float[,] gcells;
        /// <summary>
        /// photoreceptors mediation pathway calculation
        /// </summary>
        protected float[] pm;
        /// <summary>
        /// membrane potential
        /// </summary>
        protected float[] mp;
        /// <summary>
        /// sigmoid membrane potential
        /// </summary>
        protected float[] smp; //sigmoid membrane potential in time series
        /// <summary>
        /// SMP after SFA
        /// </summary>
        protected float[] sfa;
        /// <summary>
        /// spiking (>=0) or not (0)
        /// </summary>
        protected byte spike;
        /// <summary>
        /// colliding (1) or not (0)
        /// </summary>
        protected byte collision;
        #endregion

        #region CONSTRUCTOR
        /// <summary>
        /// default constructor
        /// </summary>
        public lgmd2OpenSource() { }

        /// <summary>
        /// constructor with parameters of input imgae stream
        /// </summary>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="fps"></param>
        public lgmd2OpenSource(int width/*frame width*/, int height /*frame height*/, int fps /*frames per second*/)
        {
            //stimuli property
            this.width = width;
            this.height = height;
            time_interval = 1000 / fps;
            //neural network layers and components
            photoreceptors = new int[height, width, 2];
            ons = new float[height, width, 2];
            offs = new float[height, width, 2];
            scells = new float[height, width];
            gcells = new float[height, width];
            pm = new float[2];
            mp = new float[2];
            smp = new float[2];
            sfa = new float[2];
            spike = 0;
            collision = 0;
            //neural network parameters
            Ncell = this.width * this.height;
            dc = 0.1f;
            Np = 1;
            Conv_ON = new float[2 * Np + 1, 2 * Np + 1];
            Conv_OFF = new float[2 * Np + 1, 2 * Np + 1];
            W_g = new float[2 * Np + 1, 2 * Np + 1];
            Conv_ON = makeONConvKernel();
            Conv_OFF = makeOFFConvKernel();
            groupKernel(ref W_g);
            Inh_ON = new float[height, width];
            Inh_OFF = new float[height, width];
            tau_ON = new float[2 * Np + 1];    //[0] for centre, [1] for nearest, [2] for diagonal
            tau_OFF = new float[2 * Np + 1];   //[0] for centre, [1] for nearest, [2] for diagonal
            lp_ON = new float[2 * Np + 1];   //[0] for centre, [1] for nearest, [2] for diagonal
            lp_OFF = new float[2 * Np + 1];  //[0] for centre, [1] for nearest, [2] for diagonal
            for (int i = 0; i < 2 * Np + 1; i++)
            {
                tau_ON[i] = 15 + i * 15;
                tau_OFF[i] = 60 + i * 60;
                lp_ON[i] = time_interval / (time_interval + tau_ON[i]);
                lp_OFF[i] = time_interval / (time_interval + tau_OFF[i]);
            }
            tau_PM = 90;
            lp_PM = time_interval / (time_interval + tau_PM);
            W_i_on = 1;
            W_i_on_base = 1;
            W_i_off = 0.5f;
            W_i_off_base = 0.5f;
            W_on = 1;
            W_off = 1;
            W_onoff = 1;
            Cw = 4;
            Delta_C = 0.01f;
            Cde = 0.5f;
            Tde = 15;
            Csig = 1;
            Tpm = 8;
            Tsp = 0.78f;
            Tsfa = 0.003f;
            Nsp = 0;
            Nts = 8;
            clip_point = 0.1f;
            Cspi = 4;
            tau_sfa = 800;
            hp_sfa = tau_sfa / (tau_sfa + time_interval);
        }
        #endregion

        #region METHOD
        /// <summary>
        /// Making a convolution kernel in ON pathway
        /// </summary>
        /// <returns></returns>
        protected float[,] makeONConvKernel()
        {
            float[,] mat = new float[2 * Np + 1, 2 * Np + 1];
            for (int i = -1; i < Np + 1; i++)
            {
                for (int j = -1; j < Np + 1; j++)
                {
                    if (i == 0 && j == 0)   //centre
                        mat[i + 1, j + 1] = 2;
                    else if (i == 0 || j == 0)  //nearest
                        mat[i + 1, j + 1] = 0.5f;
                    else   //diagonal
                        mat[i + 1, j + 1] = 0.25f;
                }
            }
            return mat;
        }
        /// <summary>
        /// Making a convolution kernel in OFF pathway
        /// </summary>
        /// <returns></returns>
        protected float[,] makeOFFConvKernel()
        {
            float[,] mat = new float[2 * Np + 1, 2 * Np + 1];
            for (int i = -1; i < Np + 1; i++)
            {
                for (int j = -1; j < Np + 1; j++)
                {
                    if (i == 0 && j == 0)   //centre
                        mat[i + 1, j + 1] = 1;
                    else if (i == 0 || j == 0)  //nearest
                        mat[i + 1, j + 1] = 0.25f;
                    else   //diagonal
                        mat[i + 1, j + 1] = 0.125f;
                }
            }
            return mat;
        }
        /// <summary>
        /// Make convolution kernel in Grouping layer
        /// </summary>
        /// <param name="mat"></param>
        protected void groupKernel(ref float[,] mat)
        {
            for (int i = 0; i < 2 * Np + 1; i++)
            {
                for (int j = 0; j < 2 * Np + 1; j++)
                {
                    mat[i, j] = 1 / 9f;
                }
            }
        }
        /// <summary>
        /// The simple version of first-order high-pass filter with no residual visual information
        /// </summary>
        /// <param name="pre_input"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected int HighpassFilter(byte pre_input, byte cur_input)
        {
            return cur_input - pre_input;
        }
        /// <summary>
        /// The first-order low-pass filter
        /// </summary>
        /// <param name="cur_input"></param>
        /// <param name="pre_input"></param>
        /// <param name="lp_t"></param>
        /// <returns></returns>
        protected float LowpassFilter(float cur_input, float pre_input, float lp_t)
        {
            return lp_t * cur_input + (1 - lp_t) * pre_input;
        }
        /// <summary>
        /// Spatiotemporal convolution in ON and OFF pathways
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="inputMatrix"></param>
        /// <param name="kernel"></param>
        /// <param name="cur_frame"></param>
        /// <param name="pre_frame"></param>
        /// <param name="lp_delay"></param>
        /// <returns></returns>
        protected float Convolution(int x, int y, float[,,] inputMatrix, float[,] kernel, int cur_frame, int pre_frame, float[] lp_delay)
        {
            float tmp = 0;
            int r, c;
            float lp;
            for (int i = -Np; i < Np + 1; i++)
            {
                //check border
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -Np; j < Np + 1; j++)
                {
                    //check border
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    //centre cell
                    if (i == 0 && j == 0)
                        lp = LowpassFilter(inputMatrix[r, c, cur_frame], inputMatrix[r, c, pre_frame], lp_delay[0]);
                    //nearest cells
                    else if (i == 0 || j == 0)
                        lp = LowpassFilter(inputMatrix[r, c, cur_frame], inputMatrix[r, c, pre_frame], lp_delay[1]);
                    //diagonal cells
                    else
                        lp = LowpassFilter(inputMatrix[r, c, cur_frame], inputMatrix[r, c, pre_frame], lp_delay[2]);
                    tmp += lp * kernel[i + Np, j + Np];
                }
            }
            return tmp;
        }
        /// <summary>
        /// Scale computation in G layer
        /// </summary>
        /// <returns></returns>
        protected float Scale()
        {
            float max = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (max < Math.Abs(gcells[i, j]))
                        max = Math.Abs(gcells[i, j]);
                }
            }
            return (float)(Delta_C + max * Math.Pow(Cw, -1));
        }
        /// <summary>
        /// Computation and thresholding of each Grouping cell
        /// </summary>
        /// <param name="scellvalue"></param>
        /// <param name="ce"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        protected float gCellValue(float scellvalue, float ce, float w)
        {
            float value = scellvalue * ce * (float)Math.Pow(w, -1);
            if (value * Cde >= Tde)
                return value;
            else
                return 0;
        }

        /// <summary>
        /// Override summation method in the ON and OFF pathways
        /// </summary>
        /// <param name="on_exc"></param>
        /// <param name="off_exc"></param>
        /// <returns></returns>
        protected float SupralinearSummation(float on_exc, float off_exc)
        {
            return W_on * on_exc + W_off * off_exc + W_onoff * on_exc * off_exc;
        }
        /// <summary>
        /// Computation of each Summation cell
        /// </summary>
        /// <param name="exc"></param>
        /// <param name="inh"></param>
        /// <param name="wi"></param>
        /// <returns></returns>
        protected float sCellValue(float exc, float inh, float wi)
        {
            float tmp = exc - inh * wi;
            if (tmp <= 0)
                return 0;
            else
                return tmp;
        }
        /// <summary>
        /// Half-wave rectifying in terms of onset response
        /// </summary>
        /// <param name="pre_output"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected float HRplusDC_ON(float pre_output, int cur_input)
        {
            if (cur_input >= clip_point)
                return cur_input + dc * pre_output;
            else
                return dc * pre_output;
        }
        /// <summary>
        /// Half-wave rectifying in terms of onset response
        /// </summary>
        /// <param name="pre_output"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected float HRplusDC_ON(float pre_output, float cur_input)
        {
            if (cur_input >= clip_point)
                return cur_input + dc * pre_output;
            else
                return dc * pre_output;
        }
        /// <summary>
        /// Half-wave rectifying in terms of offset response
        /// </summary>
        /// <param name="pre_output"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected float HRplusDC_OFF(float pre_output, int cur_input)
        {
            if (cur_input < clip_point)
                return Math.Abs(cur_input) + dc * pre_output;
            else
                return dc * pre_output;
        }
        /// <summary>
        /// Half-wave rectifying in terms of offset response
        /// </summary>
        /// <param name="pre_output"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected float HRplusDC_OFF(float pre_output, float cur_input)
        {
            if (cur_input < clip_point)
                return Math.Abs(cur_input) + dc * pre_output;
            else
                return dc * pre_output;
        }
        /// <summary>
        /// Convolution
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="matrix"></param>
        /// <param name="kernel"></param>
        /// <returns></returns>
        protected float Convolving(int x, int y, float[,] matrix, float[,] kernel)
        {
            float tmp = 0;
            int r, c;
            for (int i = -Np; i < Np + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -Np; j < Np + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    tmp += matrix[r, c] * kernel[i + Np, j + Np];
                }
            }
            return tmp;
        }
        /// <summary>
        /// Sigmoid transformation of membrane potential
        /// </summary>
        /// <param name="Kf"></param>
        /// <returns></returns>
        protected float SigmoidTransfer(float Kf)
        {
            return (float)Math.Pow(1 + Math.Exp(-Kf * Math.Pow(Ncell * Csig, -1)), -1);
        }
        /// <summary>
        /// Spike-Frequency Adaptation (SFA) mechanism
        /// </summary>
        /// <param name="pre_sfa"></param>
        /// <param name="pre_mp"></param>
        /// <param name="cur_mp"></param>
        /// <returns></returns>
        protected float SFA_HPF(float pre_sfa, float pre_mp, float cur_mp)
        {
            float diff_mp = cur_mp - pre_mp;
            if (diff_mp <= Tsfa)
            {
                float tmp_mp = hp_sfa * (pre_sfa + diff_mp);
                if (tmp_mp < 0.5f)
                    return 0.5f;
                else
                    return tmp_mp;
            }
            else
            {
                float tmp_mp = hp_sfa * cur_mp;
                if (tmp_mp < 0.5f)
                    return 0.5f;
                else
                    return tmp_mp;
            }
        }
        /// <summary>
        /// Spiking mechanism via exponentially mapping
        /// </summary>
        /// <param name="sfa"></param>
        /// <returns></returns>
        protected byte Spiking(float sfa)
        {
            byte spi = (byte)Math.Floor(Math.Exp(Cspi * (sfa - Tsp)));
            if (spi == 0)
                Nsp = 0;
            else
                Nsp += spi;
            return spi;
        }
        /// <summary>
        /// Detect potential collisions
        /// </summary>
        /// <returns></returns>
        protected byte loomingDetecting()
        {
            if (Nsp >= Nts)
                return 1;
            else
                return 0;
        }


        /// <summary>
        /// LGMD2 visual neural network processing
        /// </summary>
        /// <param name="img1"></param>
        /// <param name="img2"></param>
        /// <param name="t"></param>
        public void LGMD2_Processing(byte[,,] img1, byte[,,] img2, int t)
        {
            //init
            int cur_frame = t % 2;
            int pre_frame = (t - 1) % 2;
            float tmp_pm = 0;
            float tmp_sum = 0;
            float scale;

            //P layer processing
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    photoreceptors[y, x, cur_frame] = HighpassFilter(img1[y, x, 0], img2[y, x, 0]);
                    tmp_pm += Math.Abs(photoreceptors[y, x, cur_frame]);
                }
            }

            //adaptive inhibition mechanism
            pm[cur_frame] = tmp_pm / Ncell;
            pm[cur_frame] = LowpassFilter(pm[cur_frame], pm[pre_frame], lp_PM);
            W_i_off = pm[cur_frame] / Tpm;
            if (W_i_off < W_i_off_base)
                W_i_off = W_i_off_base;
            W_i_on = pm[cur_frame] / Tpm;
            if (W_i_on < W_i_on_base)
                W_i_on = W_i_on_base;

            //ON and OFF mechanisms
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    ons[y, x, cur_frame] = HRplusDC_ON(ons[y, x, pre_frame], photoreceptors[y, x, cur_frame]);
                    offs[y, x, cur_frame] = HRplusDC_OFF(offs[y, x, pre_frame], photoreceptors[y, x, cur_frame]);
                }
            }

            //spatiotemporal visual processing in partial neural networks
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Inh_ON[y, x] = Convolution(y, x, ons, Conv_ON, cur_frame, pre_frame, lp_ON);
                    Inh_OFF[y, x] = Convolution(y, x, offs, Conv_OFF, cur_frame, pre_frame, lp_OFF);
                    S_on = sCellValue(ons[y, x, cur_frame], Inh_ON[y, x], W_i_on);
                    S_off = sCellValue(offs[y, x, cur_frame], Inh_OFF[y, x], W_i_off);
                    scells[y, x] = SupralinearSummation(S_on, S_off);
                }
            }
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    gcells[y, x] = Convolving(y, x, scells, W_g);
                }
            }
            scale = Scale();
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    gcells[y, x] = gCellValue(scells[y, x], gcells[y, x], scale);
                    tmp_sum += gcells[y, x];
                }
            }

            //LGMD2 cell processing
            mp[cur_frame] = tmp_sum;
            smp[cur_frame] = SigmoidTransfer(mp[cur_frame]);
            sfa[cur_frame] = SFA_HPF(sfa[pre_frame], smp[pre_frame], smp[cur_frame]);

            //spiking and collision recognition mechanisms
            spike = Spiking(sfa[cur_frame]);
            collision = loomingDetecting();
        }
        #endregion
    }
}


