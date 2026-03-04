J2K_TRANSLATION_TEMPLATE = """You are an idiomatic code translator with expert-level knowledge of Kotlin and Java. Your task is to translate a Java function to Kotlin, using the Kotlin function signature provided.

You should aim to generate idiomatic Kotlin code, while preserving the semantics of the original Java code as much as possible. Try to maintain type safety. Replace unavailable APIs with the Kotlin standard library or the closest equivalent where possible.

Explain your reasoning concisely. You should consider inputs, outputs, control flow (conditions and loops), interactions with other functions, libraries and APIs.

Example 1:
Code:
```java
public static String numberToWord(int x) {{
    String msg;
    switch (x) {{
        case 1:
            msg = "one";
            break;
        case 2:
            msg = "two";
            break;
        default:
            msg = "other";
            break;
    }}
    return msg;
}}
```
Reasoning:
---
The "numberToWord" function takes as input an integer "x". It outputs a string that's either "one", "two", or "other". The "switch" expression branches on x: 1 -> "one", x: 2 -> "two", else -> "other".
---

Example 2:
Code:
```java
public static void printRange(int n) {{
    // inclusive (0..n)
    for (int i = 0; i <= n; i++) {{
        System.out.println(i);
    }}
    // exclusive (0 until n)
    for (int i = 0; i < n; i++) {{
        System.out.println(i);
    }}
}}
```
Reasoning:
---
The "printRange" function takes as input an integer "n". It operates two sequential loops which prints from 0 to n, the first loop using "<=" is inclusive, the second loop using "<" is exclusive.
---

Example 3:
Code:
```java
public static int max(int a, int b) {{
    if (isLarger(a, b)) {{
        return a;
    }} else {{
        return b;
    }}
}}
```
Reasoning:
---
The "max" function takes as input two integers "a" and "b" and returns the larger integer. It has an "if/else" expression to choose which number will be returned. The "isLarger" function is invoked by the "max" function to determine if integer "a" is larger than integer "b".
---

Use reasoning like this to assist your translation.

Output the reasoning in the block wrapped by '---'. Output the requested function and relevant imports in a fenced code block wrapped by '```'.
Output format:
---
<reasoning>
---
```kotlin
<code>
```

Now, translate this Java function to Kotlin, following the instructions above.
```java
{java_src_code}
```

Kotlin function signature:
{kotlin_signature}
"""



J2K_FIXUP_TEMPLATE = """You are an idiomatic code fixer with expert-level knowledge of Kotlin and Java. Your task is to fix a Kotlin function that has been partially-translated from Java.

You should aim to generate idiomatic Kotlin, while preserving the semantics of the original code as much as possible. Try to maintain type safety. Replace unavailable APIs with the Kotlin standard library or the closest equivalent where possible.

Explain your reasoning concisely. You should consider types, inputs, outputs, control flow (conditions and loops), libraries and APIs.

Example 1:
Code:
```kotlin
fun collectSquares(nums: IntArray): List<Int> {{
    val result: List<Int> = ArrayList()
    for (n in nums) {{
        (result as ArrayList).add(n * n)
    }}
    return result
}}
```
Reasoning:
---
The "collectSquares" function takes an array of integers "nums" and builds a list of their squares. Output type is List<Int>.
---

Example 2:
Code:
```kotlin
fun countZeros(numbers: List<Int>): Int {{
    var count = 0
    for (i in 0 until numbers.size()) {{
        if (numbers.get(i) == 0) {{
            count++
        }}
    }}
    return count
}}
```
Reasoning:
---
The function "countZeros" iterates through a list of integers and counts how many 0s appear. Input is List<Int>, output is an Int.
---

Example 3:
Code:
```kotlin
fun sumValues(values: List<Integer?>): Integer? {{
    var sum: Integer? = 0
    for (v in values) {{
        if (v != null) {{
            sum = sum!! + v
        }}
    }}
    return sum
}}
```
Reasoning:
---
The "sumValues" function sums a list of numbers. It takes a list named "values" as input and returns an integer as output.
---

Use reasoning like this to assist in fixing the code.

Output the reasoning in a block wrapped by '---'. Output the requested function and relevant imports in a fenced code block wrapped by '```'.
Output format:
---
<reasoning>
---
```kotlin
<code>
```

Now, fix this code where necessary, following the instructions above.
```kotlin
{kotlin_src_code}
```

Use this function signature:
{kotlin_signature}
"""