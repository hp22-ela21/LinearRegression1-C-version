/********************************************************************************
* lin_reg.h: Inneh�ller funktionalitet f�r enkel implementering av
*            maskininl�rningsmodeller baserade p� linj�r regression via
*            strukten lin_reg samt tillh�rande externa funktioner:
********************************************************************************/
#ifndef LIN_REG_H_
#define LIN_REG_H_

/* Inkluderingsdirektiv: */
#include <stdio.h>
#include <stdlib.h>

/********************************************************************************
* lin_reg: Strukt f�r implementering av maskininl�rningsmodeller baserade p�
*          linj�r regression. Tr�ningsdata passeras via pekare till arrayer
*          inneh�llande tr�ningsupps�ttningarnas in- och utdata.
********************************************************************************/
struct lin_reg
{
   const double* train_in;  /* Indata f�r tr�ningsupps�ttningar. */
   const double* train_out; /* Utdata f�r tr�ningsupps�ttningar. */
   size_t* train_order;     /* Lagrar ordningsf�ljden vid tr�ning. */
   size_t num_sets;         /* Antalet befintliga tr�ningsupps�ttningar. */
   double bias;             /* Vilov�rde (m-v�rde). */
   double weight;           /* Lutning (k-v�rde). */
};

/* Externa funktioner: */
void lin_reg_new(struct lin_reg* self);
void lin_reg_delete(struct lin_reg* self);
struct lin_reg* lin_reg_ptr_new(void);
void lin_reg_ptr_delete(struct lin_reg** self);
int lin_reg_set_training_data(struct lin_reg* self,
                              const double* train_in,
                              const double* train_out,
                              const size_t num_sets);
void lin_reg_train(struct lin_reg* self,
                   const size_t num_epochs,
                   const double learning_rate);
double lin_reg_predict(const struct lin_reg* self,
                       const double input);
void lin_reg_predict_train_in(const struct lin_reg* self,
                              FILE* ostream);
void lin_reg_predict_range(const struct lin_reg* self,
                           const double min,
                           const double max,
                           const double step,
                           FILE* ostream);

#endif /* LIN_REG_H_ */