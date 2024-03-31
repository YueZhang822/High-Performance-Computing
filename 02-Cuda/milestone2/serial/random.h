/*
 * Random number generator
 * Reference:
 *  + https://github.com/notr1ch/opentdm/blob/master/mt19937.c
 *  + http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/VERSIONS/C-LANG/mt19937-64.c
*/


void init_genrand(unsigned long s);

void init_by_array(unsigned long init_key[], int key_length);

unsigned long genrand_int32(void);

long genrand_int31(void);

double genrand_float32_full(void);

double genrand_float32_notone(void);