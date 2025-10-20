#include <iostream>
using std::cout;
using std::endl;

void MergeSortL(int *array, int size);

void MergeL(int *temp, int *part1, int *part2, int len1, int len2);

int FindPivot(int *array, int left, int right);

void PartitionL(int *array, int left, int right);

void QuickSort(int *array, int left, int right);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

void MergeSortL(int *array, int size) {
  int *temp = new int[size];
  for (int part_size = 1; part_size < size; part_size *= 2) {
    for (int j = 0; j + part_size < size; j += part_size * 2) {
      int part2_len = part_size;
      part2_len = std::min(part_size, size - j - part_size);
      MergeL(temp, &array[j], &array[j + part_size], part_size, part2_len);
      for (int i = 0; i < part_size + part2_len; i++)
        array[j + i] = temp[i];
    }
  }
  delete[] temp;
}

void MergeL(int temp[], int *part1, int *part2, int len1, int len2) {
  int i = 0, j = 0, k = 0;
  while (i < len1 && j < len2) {
    if (part1[i] <= part2[j]) {
      temp[k] = part1[i++];
    } else {
      temp[k] = part2[j++];
    }
    k++;
  }
  while (i < len1)
    temp[k++] = part1[i++];
  while (j < len2)
    temp[k++] = part2[j++];
}

void QuickSort(int *array, int left, int right) {
  if (left >= right)
    return;
  int p;
  p = FindPivot(array, left, right);
  QuickSort(array, left, p - 1);
  QuickSort(array, p + 1, right);
}

int FindPivot(int *array, int left, int right) {
  int p = array[left], i = left + 1, j = right;
  while (i <= j) {
    while (array[i] <= p && i <= j)
      i++;
    while (p <= array[j] && i <= j)
      j--;

      if (i < j)
        std::swap(array[i], array[j]);
  }
  std::swap(array[left], array[j]);
  return j;
}


int main() {
  int sort_data[20] = {2,  1,  7,  9,  6,  4,  8,  5,  3,  10,
                       12, 11, 14, 13, 16, 15, 19, 17, 18, 0};

  for (int i = 0; i < 20; i++) {
    cout << sort_data[i] << " ";
  }
  cout << endl;

  MergeSortL(sort_data, 20);
  // QuickSort(sort_data, 0, 19);

  for (int i = 0; i < 20; i++) {
    cout << sort_data[i] << " ";
  }
  cout << endl;
  return 0;
}