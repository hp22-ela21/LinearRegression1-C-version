/********************************************************************************
* main.c: Implementering av en enkel maskininl�rningsmodell baserad p� linj�r
*         regression, med tr�ningsdata definierat direkt i funktionen main.
*
*         I Windows, kompilera programkoden och skapa en k�rbar fil d�pt
*         main.exe via f�ljande kommando:
*         $ gcc main.c lin_reg.c -o main.exe -Wall
*
*         Programmet kan sedan k�ras under 10 000 epoker med en l�rhastighet
*         p� 1 % via f�ljande kommando:
*         $ main.exe
*
*         F�r att mata in antalet epoker samt l�rhastighet som skall anv�ndas
*         vid tr�ning kan f�ljande kommando anv�ndas:
*         $ main.exe <num_epochs> <learning_rate>
*
*         Som exempel, f�r att genomf�ra tr�ning under 5000 epoker med en
*         l�rhastighet p� 2 % kan f�ljande kommando anv�ndas:
*         $ main.exe 5000 0.02
********************************************************************************/
#include "lin_reg.h"

/********************************************************************************
* main: Tr�nar en maskininl�rningsmodell baserad p� linj�r regression via
*       tr�ningsdata best�ende av fem tr�ningsupps�ttningar, lagrade via var
*       sin vektor. Modellen tr�nas som default under 10 000 epoker med en
*       l�rhastighet p� 1 %. Dessa parametrar kan dock v�ljas av anv�ndaren
*       via inmatning i samband med k�rning av programmet, vilket l�ses in
*       via ing�ende argument argc samt argv.
*
*       Efter tr�ningen �r slutf�rd sker prediktion f�r samtliga insignaler
*       mellan -10 och 10 med en stegringshastighet p� 1.0. Varje insignal
*       i detta intervall skrivs ut i terminalen tillsammans med predikterad
*       utsignal.
*
*       - argc: Antalet argument som har matats in vid k�rning av programmet
*               (default = 1, vilket �r kommandot f�r att k�ra programmet).
*       - argc: Pekare till array inneh�llande samtliga inl�sta argument i
*               form av text (default = exekveringskommandot, exempelvis main).
********************************************************************************/
int main(const int argc,
         const char** argv)
{
   struct lin_reg l1;

   const double train_in[] = { 0, 1, 2, 3, 4 };
   const double train_out[] = { 2, 12, 22, 32, 42 };

   size_t num_epochs = 10000;
   double learning_rate = 0.01;

   if (argc == 3)
   {
      num_epochs = (size_t)atoi(argv[2]);
      learning_rate = atof(argv[3]);
   }

   lin_reg_new(&l1);
   lin_reg_set_training_data(&l1, train_in, train_out, 5);
   lin_reg_train(&l1, num_epochs, learning_rate);

   lin_reg_predict_range(&l1, -10, 10, 1, stdout);
   return 0;
}