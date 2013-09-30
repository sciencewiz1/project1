/**
 * Copyright (c) 2012 MIT License by 6.172 Staff
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/

// Implements the ADT specified in bitarray.h as a packed array of bits; a bit
// array containing bit_sz bits will consume roughly bit_sz/8 bytes of
// memory.


#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
 
#include <sys/types.h>

#include "./bitarray.h"


//  
//  BlockAlign(): Aligns P on the next V boundary.  
//  BlockAlignTruncate(): Aligns P on the prev V boundary.  
//    
#define BlockAlign(P,V) ((((P)) + (V-1) & (-(V))))  
#define BlockAlignTruncate(P,V) ((P) & (-(V)))  


//  
//  BlockOffset(): Calculates offset within V of P  
//    
#define BlockOffset(P,V) ((P) & (V-1))  
  
//  
//  IsBlockAligned(): Tests if P is aligned to V  
//    
#define IsBlockAligned(P,V) (BlockOffset(P, V) == 0) 


// ********************************* Types **********************************

// Concrete data type representing an array of bits.
struct bitarray {
  // The number of bits represented by this bit array.
  // Need not be divisible by 8.
  size_t bit_sz;

  // The underlying memory buffer that stores the bits in
  // packed form (8 per byte).
  char *buf;
};


// ******************** Prototypes for static functions *********************
/*
// Rotates a subarray left by an arbitrary number of bits.
//
// bit_offset is the index of the start of the subarray
// bit_length is the length of the subarray, in bits
// bit_left_amount is the number of places to rotate the
//                    subarray left
//
// The subarray spans the half-open interval
// [bit_offset, bit_offset + bit_length)
// That is, the start is inclusive, but the end is exclusive.
static void bitarray_rotate_left(bitarray_t *const bitarray,
                                 const size_t bit_offset,
                                 const size_t bit_length,
                                 const size_t bit_left_amount);

// Rotates a subarray left by one bit.
//
// bit_offset is the index of the start of the subarray
// bit_length is the length of the subarray, in bits
//
// The subarray spans the half-open interval
// [bit_offset, bit_offset + bit_length)
// That is, the start is inclusive, but the end is exclusive.
static void bitarray_rotate_left_one(bitarray_t *const bitarray,
                                     const size_t bit_offset,
                                     const size_t bit_length);
*/

// Portable modulo operation that supports negative dividends.
//
// Many programming languages define modulo in a manner incompatible with its
// widely-accepted mathematical definition.
// http://stackoverflow.com/questions/1907565/c-python-different-behaviour-of-the-modulo-operation
// provides details; in particular, C's modulo
// operator (which the standard calls a "remainder" operator) yields a result
// signed identically to the dividend e.g., -1 % 10 yields -1.
// This is obviously unacceptable for a function which returns size_t, so we
// define our own.
//
// n is the dividend and m is the divisor
//
// Returns a positive integer r = n (mod m), in the range
// 0 <= r < m.
static size_t modulo(const ssize_t n, const size_t m);

// Produces a mask which, when ANDed with a byte, retains only the
// bit_index th byte.
//
// Example: bitmask(5) produces the byte 0b00100000.
//
// (Note that here the index is counted from right
// to left, which is different from how we represent bitarrays in the
// tests.  This function is only used by bitarray_get and bitarray_set,
// however, so as long as you always use bitarray_get and bitarray_set
// to access bits in your bitarray, this reverse representation should
// not matter.
static char bitmask(const size_t bit_index);

// Declare swap
static void swap(bitarray_t * const bitarray, int start_left, int start_right, int length);

static void swap_1bit(bitarray_t * const bitarray, int start_left, int start_right);
static void swap_8bit(bitarray_t * const bitarray, int start_left, int start_right);
static void swap_16bit(bitarray_t * const bitarray, int start_left, int start_right);
static void swap_32bit(bitarray_t * const bitarray, int start_left, int start_right);
static void swap_64bit(bitarray_t * const bitarray, int start_left, int start_right);

static uint8_t bitarray_get_8bit(bitarray_t * const bitarray, int bit_index);
static uint16_t bitarray_get_16bit(bitarray_t * const bitarray, int bit_index);
static uint32_t bitarray_get_32bit(bitarray_t * const bitarray, int bit_index);
static uint64_t bitarray_get_64bit(bitarray_t * const bitarray, int bit_index);

static void bitarray_set_8bit(bitarray_t * const bitarray, int bit_index, uint8_t val);
static void bitarray_set_16bit(bitarray_t * const bitarray, int bit_index, uint16_t val);
static void bitarray_set_32bit(bitarray_t * const bitarray, int bit_index, uint32_t val);
static void bitarray_set_64bit(bitarray_t * const bitarray, int bit_index, uint64_t val);

static uint8_t bitarray_get_8bit_aligned(bitarray_t * const bitarray, int bit_index);
static uint16_t bitarray_get_16bit_aligned(bitarray_t * const bitarray, int bit_index);
static uint32_t bitarray_get_32bit_aligned(bitarray_t * const bitarray, int bit_index);
static uint64_t bitarray_get_64bit_aligned(bitarray_t * const bitarray, int bit_index);

static void bitarray_set_8bit_aligned(bitarray_t * const bitarray, int bit_index, uint8_t val);
static void bitarray_set_16bit_aligned(bitarray_t * const bitarray, int bit_index, uint16_t val);
static void bitarray_set_32bit_aligned(bitarray_t * const bitarray, int bit_index, uint32_t val);
static void bitarray_set_64bit_aligned(bitarray_t * const bitarray, int bit_index, uint64_t val);

static int DIV(int x, int y);
// ******************************* Constants ********************************
uint8_t masks8left[] = {0xFF,0xFE,0xFC,0xF8,0xF0,0xE0,0xC0,0x80,0x00};
uint8_t masks8right[] = {0xFF,0x7F,0x3F,0x1F,0x0F,0x07,0x03,0x01,0x00};
uint16_t masks16left[]={0xffff,0xfffe,0xfffc,0xfff8,0xfff0,0xffe0,0xffc0,0xff80,0xff00,0xfe00,0xfc00,0xf800,0xf000,0xe000,0xc000,0x8000};
uint16_t masks16right[]={0xffff,0x7fff,0x3fff,0x1fff,0xfff,0x7ff,0x3ff,0x1ff,0xff,0x7f,0x3f,0x1f,0xf,0x7,0x3,0x1};
uint32_t masks32left[]={0xffffffffL,0xfffffffeL,0xfffffffcL,0xfffffff8L,0xfffffff0L,0xffffffe0L,0xffffffc0L,
0xffffff80L,0xffffff00L,0xfffffe00L,0xfffffc00L,0xfffff800L,0xfffff000L,0xffffe000L,0xffffc000L,0xffff8000L,
0xffff0000L,0xfffe0000L,0xfffc0000L,0xfff80000L,0xfff00000L,0xffe00000L,0xffc00000L,0xff800000L,0xff000000L,0xfe000000L
,0xfc000000L,0xf8000000L,0xf0000000L,0xe0000000L,0xc0000000L,0x80000000L};
uint32_t masks32right[]={0xffffffffL,0x7fffffff,0x3fffffff,0x1fffffff,0xfffffff,0x7ffffff,0x3ffffff,0x1ffffff,0xffffff,
0x7fffff,0x3fffff,0x1fffff,0xfffff,0x7ffff,0x3ffff,0x1ffff,0xffff,0x7fff,0x3fff,0x1fff,0xfff,0x7ff,0x3ff,0x1ff,0xff,0x7f,
0x3f,0x1f,0xf,0x7,0x3,0x1};
// ******************************* Functions ********************************

bitarray_t *bitarray_new(const size_t bit_sz) {
  // Allocate an underlying buffer of ceil(bit_sz/8) bytes.
  char *const buf = calloc(1, bit_sz / 8 + ((bit_sz % 8 == 0) ? 0 : 1));
  if (buf == NULL) {
    return NULL;
  }

  // Allocate space for the struct.
  bitarray_t *const bitarray = malloc(sizeof(struct bitarray));
  if (bitarray == NULL) {
    free(buf);
    return NULL;
  }

  bitarray->buf = buf;
  bitarray->bit_sz = bit_sz;
  return bitarray;
}

void bitarray_free(bitarray_t *const bitarray) {
  if (bitarray == NULL) {
    return;
  }
  free(bitarray->buf);
  bitarray->buf = NULL;
  free(bitarray);
}

size_t bitarray_get_bit_sz(const bitarray_t *const bitarray) {
  return bitarray->bit_sz;
}

bool bitarray_get(const bitarray_t *const bitarray, const size_t bit_index) {
  assert(bit_index < bitarray->bit_sz);

  // We're storing bits in packed form, 8 per byte.  So to get the nth
  // bit, we want to look at the (n mod 8)th bit of the (floor(n/8)th)
  // byte.
  //
  // In C, integer division is floored explicitly, so we can just do it to
  // get the byte; we then bitwise-and the byte with an appropriate mask
  // to produce either a zero byte (if the bit was 0) or a nonzero byte
  // (if it wasn't).  Finally, we convert that to a boolean.
  return (bitarray->buf[bit_index / 8] & bitmask(bit_index)) ?
             true : false;
}

void bitarray_set(bitarray_t *const bitarray,
                  const size_t bit_index,
                  const bool value) {
  assert(bit_index < bitarray->bit_sz);

  // We're storing bits in packed form, 8 per byte.  So to set the nth
  // bit, we want to set the (n mod 8)th bit of the (floor(n/8)th) byte.
  //
  // In C, integer division is floored explicitly, so we can just do it to
  // get the byte; we then bitwise-and the byte with an appropriate mask
  // to clear out the bit we're about to set.  We bitwise-or the result
  // with a byte that has either a 1 or a 0 in the correct place.
  bitarray->buf[bit_index / 8] =
      (bitarray->buf[bit_index / 8] & ~bitmask(bit_index)) |
           (value ? bitmask(bit_index) : 0);
}

void bitarray_rotate(bitarray_t *const bitarray,
                     const size_t bit_offset,
                     const size_t bit_length,
                     const ssize_t bit_right_amount) {
  assert(bit_offset + bit_length <= bitarray->bit_sz);

  if (bit_length == 0) {
    return;
  }

  // Convert a rotate left or right to a left rotate only, and eliminate
  // multiple full rotations.

  // The amount of left rotation
  int d = modulo(-bit_right_amount, bit_length);

  // If we are rotating by 0 slots then just stop
  if (d == 0) {
    return;
  }

  // Initialize rotation variables
  // i indicates the number of bits we still need to swap from the left side
  int i = d;
  // j indicates the number of bits we still need to swap from the right side
  int j = bit_length - d;

  while (i != j) {
    if (i < j) {
      // Left part is bigger than the right part, swap left to its final position
      // in the right side
      // e.g: A[0..7] left-rotate by 3; swap A[0..2] with A[5..7]
      swap(bitarray, bit_offset + d - i, bit_offset + d + j - i, i);
      j -= i;
    } else {
      // right part is bigger than the left part, swap right to its final position
      // in the left side
      // e.g: A[0..7] left-rotate by 3; swap A[0..2] with A[5..7]
      swap(bitarray, bit_offset + d - i, bit_offset + d, j);
      i -= j;
    }
  }

  // Left part and right part are of the same size, we just swap them directly
  // e.g: A[0..7] left-rotate by 4; swap A[0..3] with A[4..7]
  swap(bitarray, bit_offset + d - i, bit_offset + d, i);

  /*bitarray_rotate_left(bitarray, bit_offset, bit_length,
           modulo(-bit_right_amount, bit_length));*/
}


// Swap the bit elements of bitarray[start_left...start_left + length + 1] with
// the bit elements of bitarray[start_right...start_right + length - 1]
inline void swap(bitarray_t * const bitarray, int start_left, int start_right, int length) {
  int counter = 0;
  /*while (length >= 64) {
    swap_64bit(bitarray, start_left + counter, start_right + counter);
    length -= 64;
    counter += 64;
  }
 
  while (length >= 32) {
    swap_32bit(bitarray, start_left + counter, start_right + counter);
    length -= 32;
    counter += 32;
  }*/
  
  while (length >= 16) {
    swap_16bit(bitarray, start_left + counter, start_right + counter);
    length -= 16;
    counter += 16;
  }
  
  while (length >= 8) {
    swap_8bit(bitarray, start_left + counter, start_right + counter);
    length -= 8;
    counter += 8;
  }
  
  while (length > 0) {
    swap_1bit(bitarray, start_left + counter, start_right + counter);
    length -= 1;
    counter += 1;
  }  
} 

inline static void swap_1bit(bitarray_t * const bitarray, int start_left, int start_right) {
  bool current_left = bitarray_get(bitarray, start_left);
  bitarray_set(bitarray, start_left, bitarray_get(bitarray, start_right));
  bitarray_set(bitarray, start_right, current_left);
}

inline static void swap_8bit(bitarray_t * const bitarray, int start_left, int start_right) {   
  uint8_t temp_left;
  uint8_t temp_right;
  int leftAligned = IsBlockAligned(start_left, 8);
  
  temp_right = bitarray_get_8bit(bitarray, start_right);
  if (leftAligned) {
    temp_left = bitarray_get_8bit_aligned(bitarray, start_left);
    bitarray_set_8bit_aligned(bitarray, start_left, temp_right);
  } else {
    temp_left = bitarray_get_8bit(bitarray, start_left);
    bitarray_set_8bit(bitarray, start_left, temp_right);
  }
  bitarray_set_8bit(bitarray, start_right, temp_left);
}

inline static void swap_16bit(bitarray_t * const bitarray, int start_left, int start_right) {
  uint16_t temp_left;
  uint16_t temp_right;
  int leftAligned = IsBlockAligned(start_left, 16);
  
  temp_right = bitarray_get_16bit(bitarray, start_right);
  if (leftAligned) {
    temp_left = bitarray_get_16bit_aligned(bitarray, start_left);
    bitarray_set_16bit_aligned(bitarray, start_left, temp_right);
  } else {
    temp_left = bitarray_get_16bit(bitarray, start_left);
    bitarray_set_16bit(bitarray, start_left, temp_right);
  }
  bitarray_set_16bit(bitarray, start_right, temp_left);
}

inline static void swap_32bit(bitarray_t * const bitarray, int start_left, int start_right) {
  uint32_t temp_left;
  uint32_t temp_right;
  int leftAligned = IsBlockAligned(start_left, 32);
    
  temp_right = bitarray_get_32bit(bitarray, start_right);
  if (leftAligned) {
    temp_left = bitarray_get_32bit_aligned(bitarray, start_left);
    bitarray_set_32bit_aligned(bitarray, start_left, temp_right);
  } else {
    temp_left = bitarray_get_32bit(bitarray, start_left);
    bitarray_set_32bit(bitarray, start_left, temp_right);
  }
  bitarray_set_32bit(bitarray, start_right, temp_left);
}

inline static void swap_64bit(bitarray_t * const bitarray, int start_left, int start_right) {
  uint64_t temp_left;
  uint64_t temp_right;
  int leftAligned = IsBlockAligned(start_left, 64);
  
  temp_right = bitarray_get_64bit(bitarray, start_right);
  if (leftAligned) {
    temp_left = bitarray_get_64bit_aligned(bitarray, start_left);
    bitarray_set_64bit_aligned(bitarray, start_left, temp_right);
  } else {
    temp_left = bitarray_get_64bit(bitarray, start_left);
    bitarray_set_64bit(bitarray, start_left, temp_right);
  }
  bitarray_set_64bit(bitarray, start_right, temp_left);
}
 
inline static uint8_t bitarray_get_8bit(bitarray_t * const bitarray, int bit_index) {
  int v = DIV(bit_index,8);
  uint8_t * buf8bit = (uint8_t *) (bitarray->buf);
  uint8_t left = *(buf8bit + v);
  uint8_t right = *(buf8bit + v + 1);
  uint8_t partialIdx = bit_index & 7;
  uint8_t partialLeft = masks8left[partialIdx] & left;
  uint8_t partialRight = masks8right[8 - partialIdx] & right;
  //uint8_t partialLeft = (0xFF << partialIdx) & left;
  //uint8_t partialRight = (0xFF >> (8 - partialIdx)) & right;
  return (partialLeft >> partialIdx) | (partialRight << (8 - partialIdx));  
}

inline static uint16_t bitarray_get_16bit(bitarray_t * const bitarray, int bit_index) {
  int v = DIV(bit_index,16);
  uint16_t * buf16bit = (uint16_t *) (bitarray->buf);
  uint16_t left = *(buf16bit + v);
  uint16_t right = *(buf16bit + v + 1);
  uint16_t partialIdx = bit_index & 15;
  uint16_t partialLeft = masks16left[partialIdx] & left;
  uint16_t partialRight = masks16right[16 - partialIdx] & right;
  //uint16_t partialLeft = (0xFFFF << partialIdx) & left;
  //uint16_t partialRight = (0xFFFF >> (16 - partialIdx)) & right;
  return (partialLeft >> partialIdx) | (partialRight << (16 - partialIdx));
}

inline static uint32_t bitarray_get_32bit(bitarray_t * const bitarray, int bit_index) {
  uint32_t * buf32bit = (uint32_t *) (bitarray->buf);
  uint32_t left = *(buf32bit + bit_index/32);
  uint32_t right = *(buf32bit + bit_index/32 + 1);
  uint32_t partialIdx = bit_index % 32;
  uint32_t partialLeft = masks32left[partialIdx] & left;
  uint32_t partialRight = masks32right[32 - partialIdx] & right;
  //uint32_t partialLeft = (0xFFFFFFFF << partialIdx) & left;
  //uint32_t partialRight = (0xFFFFFFFF >> (32 - partialIdx)) & right;
  return (partialLeft >> partialIdx) | (partialRight << (32 - partialIdx));
}

inline static uint64_t bitarray_get_64bit(bitarray_t * const bitarray, int bit_index) {
  uint64_t * buf64bit = (uint64_t *) (bitarray->buf + bit_index/8);
  uint64_t left = *(buf64bit + bit_index/64);
  uint64_t right = *(buf64bit + bit_index/64 + 1);
  uint64_t partialIdx = bit_index % 64;
  uint64_t partialLeft = (0xFFFFFFFFFFFFFFFF << partialIdx) & left;
  uint64_t partialRight = (0xFFFFFFFFFFFFFFFF >> (64 - partialIdx)) & right;
  return (partialLeft >> partialIdx) | (partialRight << (64 - partialIdx));
}


inline static void bitarray_set_8bit(bitarray_t * const bitarray, int bit_index, uint8_t val) {
  int v = DIV(bit_index,8);
  uint8_t * buf8bit = (uint8_t *) (bitarray->buf);
  uint8_t partialIdx = bit_index & 7;
  uint8_t partialLeft = (0xFF >> partialIdx) & val;  
  uint8_t partialRight = val >> (8 - partialIdx);
  uint8_t left = ((0xFF >> (8 - partialIdx)) & (*(buf8bit + v ))) | (partialLeft << partialIdx);
  uint8_t right = ((0xFF << partialIdx) & (*(buf8bit + v + 1))) | partialRight;
  *(buf8bit + v) = left;
  *(buf8bit + v + 1) = right;  
}

inline static void bitarray_set_16bit(bitarray_t * const bitarray, int bit_index, uint16_t val) {
  int v = DIV(bit_index,16);
  uint16_t * buf16bit = (uint16_t *) (bitarray->buf);
  uint16_t partialIdx = bit_index & 15;
  uint16_t partialLeft = (0xFFFF >> partialIdx) & val;  
  uint16_t partialRight = val >> (16 - partialIdx);
  uint16_t left = ((0xFFFF >> (16 - partialIdx)) & (*(buf16bit + v))) | (partialLeft << partialIdx);
  uint16_t right = ((0xFFFF << partialIdx) & (*(buf16bit + v + 1))) | partialRight;
  *(buf16bit + v) = left;
  *(buf16bit + v + 1) = right;  
}

inline static void bitarray_set_32bit(bitarray_t * const bitarray, int bit_index, uint32_t val) {
  uint32_t * buf32bit = (uint32_t *) (bitarray->buf);
  uint32_t partialIdx = bit_index % 32;
  uint32_t partialLeft = (0xFFFFFFFF >> partialIdx) & val;  
  uint32_t partialRight = val >> (32 - partialIdx);
  uint32_t left = ((0xFFFFFFFF >> (32 - partialIdx)) & (*(buf32bit + bit_index/32 ))) | (partialLeft << partialIdx);
  uint32_t right = ((0xFFFFFFFF << partialIdx) & (*(buf32bit + bit_index/32 + 1))) | partialRight;
  *(buf32bit + bit_index/32) = left;
  *(buf32bit + bit_index/32 + 1) = right;  
}

inline static void bitarray_set_64bit(bitarray_t * const bitarray, int bit_index, uint64_t val) {
  uint64_t * buf64bit = (uint64_t *) (bitarray->buf);
  uint64_t partialIdx = bit_index % 64;
  uint64_t partialLeft = (0xFFFFFFFFFFFFFFFF >> partialIdx) & val;  
  uint64_t partialRight = val >> (64 - partialIdx);
  uint64_t left = ((0xFFFFFFFFFFFFFFFF >> (64 - partialIdx)) & (*(buf64bit + bit_index/64 ))) | (partialLeft << partialIdx);
  uint64_t right = ((0xFFFFFFFFFFFFFFFF << partialIdx) & (*(buf64bit + bit_index/64 + 1))) | partialRight;
  *(buf64bit + bit_index/64) = left;
  *(buf64bit + bit_index/64 + 1) = right;
}


// Assume that it's aligned
inline static uint8_t bitarray_get_8bit_aligned(bitarray_t * const bitarray, int bit_index) {
  uint8_t * buf8bit = (uint8_t *) (bitarray->buf) + DIV(bit_index,8);
  return *buf8bit;
}

inline static uint16_t bitarray_get_16bit_aligned(bitarray_t * const bitarray, int bit_index) {
  uint16_t * buf16bit = (uint16_t *) (bitarray->buf) + DIV(bit_index,16);
  return *buf16bit;  
}

inline static uint32_t bitarray_get_32bit_aligned(bitarray_t * const bitarray, int bit_index) {
  uint32_t * buf32bit = (uint32_t *) (bitarray->buf) + bit_index/32;
  return *buf32bit;
}

inline static uint64_t bitarray_get_64bit_aligned(bitarray_t * const bitarray, int bit_index) {
  uint64_t * buf64bit = (uint64_t *) (bitarray->buf) + bit_index/64;
  return *buf64bit;
}


static void bitarray_set_8bit_aligned(bitarray_t * const bitarray, int bit_index, uint8_t val) {
  uint8_t * buf8bit = (uint8_t *) (bitarray->buf) + bit_index/8;
  *buf8bit = val;
}

static void bitarray_set_16bit_aligned(bitarray_t * const bitarray, int bit_index, uint16_t val) {
  uint16_t * buf16bit = (uint16_t *) (bitarray->buf) + bit_index/16;
  *buf16bit = val;
}

static void bitarray_set_32bit_aligned(bitarray_t * const bitarray, int bit_index, uint32_t val) {
  uint32_t * buf32bit = (uint32_t *) (bitarray->buf) + bit_index/32;
  *buf32bit = val;
}

static void bitarray_set_64bit_aligned(bitarray_t * const bitarray, int bit_index, uint64_t val) {
  uint64_t * buf64bit = (uint64_t *) (bitarray->buf) + bit_index/64;
  *buf64bit = val;
}

static size_t modulo(const ssize_t n, const size_t m) {
  const ssize_t signed_m = (ssize_t)m;
  assert(signed_m > 0);
  const ssize_t result = ((n % signed_m) + signed_m) % signed_m;
  assert(result >= 0);
  return (size_t)result;
}

static char bitmask(const size_t bit_index) {
  return 1 << (bit_index % 8);
}
static int DIV(int x, int y)
{
	int n = 0;
	while (x > y) {
		x -= y;
		n++;
	}
	return n; // truncate fraction
}
