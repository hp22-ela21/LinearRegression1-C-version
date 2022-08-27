/********************************************************************************
* lin_reg.c: Definition av externa funktioner för ämnade för strukten lin_reg,
*            som används för implementering av enkla maskininlärningsmodeller
*            som baseras på linjär regression.
********************************************************************************/
#include "lin_reg.h"

/* Statiska funktioner: */
static void lin_reg_shuffle(struct lin_reg* self);
static void lin_reg_optimize(struct lin_reg* self,
                             const double input,
                             const double reference,
                             const double learning_rate);
static double get_random(void);
static inline size_t* uint_ptr_new(size_t size);

/********************************************************************************
* lin_reg_new: Initierar angiven regressionsmodell. Modellens bias och vikt
*              tilldelas randomiserade startvärden mellan 0.0 - 1.0.
* 
*              - self: Pekare till regressionsmodellen.
********************************************************************************/
void lin_reg_new(struct lin_reg* self)
{
   self->train_in = 0;
   self->train_out = 0;
   self->train_order = 0;
   self->num_sets = 0;
   self->bias = get_random();
   self->weight = get_random();
   return;
}

/********************************************************************************
* lin_reg_delete: Nollställer angiven regressionsmodell.
*
*                 - self: Pekare till regressionsmodellen.
********************************************************************************/
void lin_reg_delete(struct lin_reg* self)
{
   self->train_in = 0;
   self->train_out = 0;

   free(self->train_order);
   self->train_order = 0;

   self->num_sets = 0;
   self->bias = 0;
   self->weight = 0;
   return;
}

/********************************************************************************
* lin_reg_ptr_new: Returnerar pekare till en ny heapallokerad regressionsmodell. 
*                  Modellens bias och vikt tilldelas randomiserade startvärden 
*                  mellan 0.0 - 1.0.
********************************************************************************/
struct lin_reg* lin_reg_ptr_new(void)
{
   struct lin_reg* self = (struct lin_reg*)malloc(sizeof(struct lin_reg));
   if (!self) return 0;
   lin_reg_new(self);
   return self;
}

/********************************************************************************
* lin_reg_ptr_delete: Raderar heapallokerad regressionsmodell och sätter
*                     motsvarande pekare till null.
* 
*                     - self: Adressen till pekaren som pekar på den 
*                             heapallokerade regressionsmodellen.
********************************************************************************/
void lin_reg_ptr_delete(struct lin_reg** self)
{
   lin_reg_delete(*self);
   free(*self);
   *self = 0;
   return;
}

/********************************************************************************
* lin_reg_set_training_data: Läser in träningsdata för angiven regressionsmodell 
*                            via passerad in- och utdata, tillsammans med att 
*                            index för respektive träningsuppsättning lagras.
*
*                            - self     : Pekare till regressionsmodellen.
*                            - train_in : Pekare till array innehållande indata.
*                            - train_out: Pekare till array innehållande utdata.
*                            - num_sets : Antalet träningsuppsättningar som
*                                         förekommer i passerad träningsdata.
********************************************************************************/
int lin_reg_set_training_data(struct lin_reg* self,
                              const double* train_in,
                              const double* train_out,
                              const size_t num_sets)
{
   self->train_in = train_in;
   self->train_out = train_out;
   self->train_order = uint_ptr_new(num_sets);

   if (!self->train_order)
   {
      self->num_sets = 0;
      return 1;
   }
   else
   {
      self->num_sets = num_sets;

      for (size_t i = 0; i < self->num_sets; ++i)
      {
         self->train_order[i] = i;
      }

      return 0;
   }
}

/********************************************************************************
* lin_reg_train: Tränar angiven regressionsmodell med befintlig träningsdata
*                under angivet antal epoker samt angiven lärhastighet. I början 
*                av varje epok randomiseras ordningen på träningsuppsättningarna 
*                för att undvika modellen blir för bekant med träningsdatan.
*
*                För varje träningsuppsättning sker en prediktion via aktuell 
*                indata. Det predikterade värdet jämförs mot aktuellt 
*                referensvärde för att beräkna aktuell avvikelse. Modellens 
*                parametrar justeras därefter.
*
*                - self         : Pekare till regressionsmodellen.
*                - num_epochs   : Antalet omgångar träning som skall genomföras.
*                - learning_rate: Lärhastigheten, som avgör hur stor andel av 
*                                 uppmätt avvikelse som modellens parametrar 
*                                 justeras med.
********************************************************************************/
void lin_reg_train(struct lin_reg* self,
                   const size_t num_epochs,
                   const double learning_rate)
{
   for (size_t i = 0; i < num_epochs; ++i)
   {
      lin_reg_shuffle(self);

      for (size_t j = 0; j < self->num_sets; ++j)
      {
         const size_t k = self->train_order[j];
         lin_reg_optimize(self, self->train_in[k], self->train_out[k], learning_rate);
      }
   }

   return;
}


/********************************************************************************
* lin_reg_predict: Genomför prediktion med angiven regressionsmodell för
*                  angiven insignal och returnerar resultatet.
* 
*                  - self : Pekare till regressionsmodellen.
*                  - input: Insignal som prediktion skall genomföras utefter.
********************************************************************************/
double lin_reg_predict(const struct lin_reg* self,
                       const double input)
{
   return self->weight * input + self->bias;
}

/********************************************************************************
* lin_reg_predict_train_in: Genomför prediktion med angiven regressionsmodell 
*                           via indata från befintliga träningsuppsättningar.
*                           Varje insignal samt motsvarande predikterad utsignal
*                           skrivs ut via angiven utström, där standardutenheten 
*                           stdout används som default för utskrift i terminalen.
*
*                           - self   : Pekare till regressionsmodellen.
*                           - ostream: Pekare till utström för uskrift
*                                      (default = stdout).
********************************************************************************/
void lin_reg_predict_train_in(const struct lin_reg* self,
                              FILE* ostream)
{
   const double threshold = 0.01;
   const double* end = 0;

   if (!self->num_sets)
   {
      fprintf(stderr, "Training data missing!\n\n");
      return;
   }

   end = self->train_in + self->num_sets - 1;
   if (!ostream) ostream = stdout;

   fprintf(ostream, "--------------------------------------------------------------------------------\n");

   for (const double* i = self->train_in; i < self->train_in + self->num_sets; ++i)
   {
      const double prediction = self->weight * (*i) + self->bias;

      fprintf(ostream, "Input: %g\n", *i);

      if (prediction > -threshold && prediction < threshold)
      {
         fprintf(ostream, "Predicted output: %g\n", 0.0);
      }
      else
      {
         fprintf(ostream, "Predicted output: %g\n", prediction);
      }

      if (i < end) fprintf(ostream, "\n");
   }

   fprintf(ostream, "--------------------------------------------------------------------------------\n\n");
   return;
}

/********************************************************************************
* lin_reg_predict_range: Genomför prediktion med angiven regressionsmodell för
*                        datapunkter inom intervallet mellan angivet min- och 
*                        maxvärde [min, max] med angiven stegringshastighet.
*
*                        Varje insignal inom intervallet skrivs ut tillsammans 
*                        med motsvarande predikterat värde via angiven utström, 
*                        där standardutenheten stdout används som default för 
*                        utskrift i terminalen.
*
*                        - self   : Pekare till regressionsmodellen.
*                        - min    : Minvärde för datatpunkter som skall testas.
*                        - max    : Maxvärde för datatpunkter som skall testas.
*                        - step   : Stegringshastigheten, dvs. differensen mellan
*                                   varje datapunkt som skall testas.
*                        - ostream: Pekare till angiven utström för utskrift 
*                                   (default = stdout).
********************************************************************************/
void lin_reg_predict_range(const struct lin_reg* self,
                           const double min,
                           const double max,
                           const double step,
                           FILE* ostream)
{
   const double threshold = 0.01;

   if (min >= max)
   {
      fprintf(stderr, "Error: Minimum input value cannot be higher or equal to maximum input value!\n\n");
      return;
   }

   if (!ostream) ostream = stdout;

   fprintf(ostream, "--------------------------------------------------------------------------------\n");

   for (double i = min; i <= max; ++i)
   {
      const double prediction = self->weight * i + self->bias;

      fprintf(ostream, "Input: %g\n", i);

      if (prediction > -threshold && prediction < threshold)
      {
         fprintf(ostream, "Predicted output: %g\n", 0.0);
      }
      else
      {
         fprintf(ostream, "Predicted output: %g\n", prediction);
      }

      if (i < max) fprintf(ostream, "\n");
   }

   fprintf(ostream, "--------------------------------------------------------------------------------\n\n");
   return;
}

/********************************************************************************
* lin_reg_shuffle: Randomiserar den inbördes ordningsföljden på befintliga 
*                  träningsuppsättningar för angiven regressionsmodell, vilket 
*                  genomförs för att modellen inte skall bli för bekant med
*                  träningsdatan.
* 
*                  - self: Pekare till regressionsmodellen.
********************************************************************************/
static void lin_reg_shuffle(struct lin_reg* self)
{
   for (size_t i = 0; i < self->num_sets; ++i)
   {
      const size_t r = rand() % self->num_sets;
      const size_t temp = self->train_order[i];
      self->train_order[i] = self->train_order[r];
      self->train_order[r] = temp;
   }

   return;
}

/********************************************************************************
* lin_reg_optimize: Beräknar aktuell avvikelse för angiven regressionsmodell
*                   och justerar modellens parametrar därefter.
*
*                   - self   : Pekare till regressionsmodellen.
*                   - input        : Insignal som prediktion skall genomföras med.
*                   - reference    : Referensvärde från träningsdatan, som utgör
*                                  det värde som modellen önskas prediktera.
*                   - learning_rate: Modellens lärhastighet, som avgör hur mycket
*                                  modellens parametrar justeras vid avvikelse.
********************************************************************************/
static void lin_reg_optimize(struct lin_reg* self,
                             const double input,
                             const double reference,
                             const double learning_rate)
{
   const double prediction = self->weight * input + self->bias;
   const double deviation = reference - prediction;
   const double change_rate = deviation * learning_rate;

   self->bias += change_rate;
   self->weight += change_rate * input;
   return;
}

/********************************************************************************
* get_random: Returnerar ett randomiserat flyttal mellan 0.0 - 1.0.
********************************************************************************/
static double get_random(void)
{
   return rand() / (double)RAND_MAX;
}

/********************************************************************************
* uint_ptr_new: Returnerar en pekare till ett heapallokerat fält som rymmer
*               angivet antal osignerade heltal.
* 
*               - size: Storleken på det heapallokerade fältet, dvs. antalet
*                       osignerade heltal det skall rymma.
********************************************************************************/
static inline size_t* uint_ptr_new(size_t size)
{
   return (size_t*)malloc(sizeof(size_t) * size);
}