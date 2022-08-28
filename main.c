/********************************************************************************
* main.c: Implementering av en enkel maskininlärningsmodell baserad på linjär
*         regression, med träningsdata definierat direkt i funktionen main.
*
*         I Windows, kompilera programkoden och skapa en körbar fil döpt
*         main.exe via följande kommando:
*         $ gcc main.c lin_reg.c -o main.exe -Wall
*
*         Programmet kan sedan köras under 10 000 epoker med en lärhastighet
*         på 1 % via följande kommando:
*         $ main.exe
*
*         För att mata in antalet epoker samt lärhastighet som skall användas
*         vid träning kan följande kommando användas:
*         $ main.exe <num_epochs> <learning_rate>
*
*         Som exempel, för att genomföra träning under 5000 epoker med en
*         lärhastighet på 2 % kan följande kommando användas:
*         $ main.exe 5000 0.02
********************************************************************************/
#include "lin_reg.h"

/********************************************************************************
* main: Tränar en maskininlärningsmodell baserad på linjär regression via
*       träningsdata bestående av fem träningsuppsättningar, lagrade via var
*       sin vektor. Modellen tränas som default under 10 000 epoker med en
*       lärhastighet på 1 %. Dessa parametrar kan dock väljas av användaren
*       via inmatning i samband med körning av programmet, vilket läses in
*       via ingående argument argc samt argv.
*
*       Efter träningen är slutförd sker prediktion för samtliga insignaler
*       mellan -10 och 10 med en stegringshastighet på 1.0. Varje insignal
*       i detta intervall skrivs ut i terminalen tillsammans med predikterad
*       utsignal.
*
*       - argc: Antalet argument som har matats in vid körning av programmet
*               (default = 1, vilket är kommandot för att köra programmet).
*       - argc: Pekare till array innehållande samtliga inlästa argument i
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