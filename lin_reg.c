/********************************************************************************
* lin_reg.c: Definition av externa funktioner f�r �mnade f�r strukten lin_reg,
*            som anv�nds f�r implementering av enkla maskininl�rningsmodeller
*            som baseras p� linj�r regression.
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
*              tilldelas randomiserade startv�rden mellan 0.0 - 1.0.
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
* lin_reg_delete: Nollst�ller angiven regressionsmodell.
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
*                  Modellens bias och vikt tilldelas randomiserade startv�rden 
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
* lin_reg_ptr_delete: Raderar heapallokerad regressionsmodell och s�tter
*                     motsvarande pekare till null.
* 
*                     - self: Adressen till pekaren som pekar p� den 
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
* lin_reg_set_training_data: L�ser in tr�ningsdata f�r angiven regressionsmodell 
*                            via passerad in- och utdata, tillsammans med att 
*                            index f�r respektive tr�ningsupps�ttning lagras.
*
*                            - self     : Pekare till regressionsmodellen.
*                            - train_in : Pekare till array inneh�llande indata.
*                            - train_out: Pekare till array inneh�llande utdata.
*                            - num_sets : Antalet tr�ningsupps�ttningar som
*                                         f�rekommer i passerad tr�ningsdata.
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
* lin_reg_train: Tr�nar angiven regressionsmodell med befintlig tr�ningsdata
*                under angivet antal epoker samt angiven l�rhastighet. I b�rjan 
*                av varje epok randomiseras ordningen p� tr�ningsupps�ttningarna 
*                f�r att undvika modellen blir f�r bekant med tr�ningsdatan.
*
*                F�r varje tr�ningsupps�ttning sker en prediktion via aktuell 
*                indata. Det predikterade v�rdet j�mf�rs mot aktuellt 
*                referensv�rde f�r att ber�kna aktuell avvikelse. Modellens 
*                parametrar justeras d�refter.
*
*                - self         : Pekare till regressionsmodellen.
*                - num_epochs   : Antalet omg�ngar tr�ning som skall genomf�ras.
*                - learning_rate: L�rhastigheten, som avg�r hur stor andel av 
*                                 uppm�tt avvikelse som modellens parametrar 
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
* lin_reg_predict: Genomf�r prediktion med angiven regressionsmodell f�r
*                  angiven insignal och returnerar resultatet.
* 
*                  - self : Pekare till regressionsmodellen.
*                  - input: Insignal som prediktion skall genomf�ras utefter.
********************************************************************************/
double lin_reg_predict(const struct lin_reg* self,
                       const double input)
{
   return self->weight * input + self->bias;
}

/********************************************************************************
* lin_reg_predict_train_in: Genomf�r prediktion med angiven regressionsmodell 
*                           via indata fr�n befintliga tr�ningsupps�ttningar.
*                           Varje insignal samt motsvarande predikterad utsignal
*                           skrivs ut via angiven utstr�m, d�r standardutenheten 
*                           stdout anv�nds som default f�r utskrift i terminalen.
*
*                           - self   : Pekare till regressionsmodellen.
*                           - ostream: Pekare till utstr�m f�r uskrift
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
* lin_reg_predict_range: Genomf�r prediktion med angiven regressionsmodell f�r
*                        datapunkter inom intervallet mellan angivet min- och 
*                        maxv�rde [min, max] med angiven stegringshastighet.
*
*                        Varje insignal inom intervallet skrivs ut tillsammans 
*                        med motsvarande predikterat v�rde via angiven utstr�m, 
*                        d�r standardutenheten stdout anv�nds som default f�r 
*                        utskrift i terminalen.
*
*                        - self   : Pekare till regressionsmodellen.
*                        - min    : Minv�rde f�r datatpunkter som skall testas.
*                        - max    : Maxv�rde f�r datatpunkter som skall testas.
*                        - step   : Stegringshastigheten, dvs. differensen mellan
*                                   varje datapunkt som skall testas.
*                        - ostream: Pekare till angiven utstr�m f�r utskrift 
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
* lin_reg_shuffle: Randomiserar den inb�rdes ordningsf�ljden p� befintliga 
*                  tr�ningsupps�ttningar f�r angiven regressionsmodell, vilket 
*                  genomf�rs f�r att modellen inte skall bli f�r bekant med
*                  tr�ningsdatan.
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
* lin_reg_optimize: Ber�knar aktuell avvikelse f�r angiven regressionsmodell
*                   och justerar modellens parametrar d�refter.
*
*                   - self   : Pekare till regressionsmodellen.
*                   - input        : Insignal som prediktion skall genomf�ras med.
*                   - reference    : Referensv�rde fr�n tr�ningsdatan, som utg�r
*                                  det v�rde som modellen �nskas prediktera.
*                   - learning_rate: Modellens l�rhastighet, som avg�r hur mycket
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
* uint_ptr_new: Returnerar en pekare till ett heapallokerat f�lt som rymmer
*               angivet antal osignerade heltal.
* 
*               - size: Storleken p� det heapallokerade f�ltet, dvs. antalet
*                       osignerade heltal det skall rymma.
********************************************************************************/
static inline size_t* uint_ptr_new(size_t size)
{
   return (size_t*)malloc(sizeof(size_t) * size);
}