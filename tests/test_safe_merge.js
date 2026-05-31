const assert = require('assert');

// Simulate the safeMerge function to test its logic
function safeMerge(target, source) {
    for (const key in source) {
        if (!Object.prototype.hasOwnProperty.call(source, key)) continue;
        if (key === '__proto__' || key === 'constructor' || key === 'prototype') continue;

        const sourceVal = source[key];
        const targetVal = target[key];

        if (typeof sourceVal === 'object' && sourceVal !== null && !Array.isArray(sourceVal)) {
            if (typeof targetVal !== 'object' || targetVal === null || Array.isArray(targetVal)) {
                target[key] = {};
            }
            safeMerge(target[key], sourceVal);
        } else {
            target[key] = sourceVal;
        }
    }
    return target;
}

console.log("Running safeMerge tests...");

// Test 1: Basic merging
let target1 = { a: 1, b: { c: 2 } };
let source1 = { b: { d: 3 }, e: 4 };
let result1 = safeMerge(target1, source1);
assert.deepStrictEqual(result1, { a: 1, b: { c: 2, d: 3 }, e: 4 }, "Basic merge failed");
console.log("✓ Basic merge passed");

// Test 2: Prototype Pollution via __proto__
let target2 = {};
let maliciousPayload1 = JSON.parse('{"__proto__": {"polluted": true}}');
safeMerge(target2, maliciousPayload1);
assert.strictEqual({}.polluted, undefined, "Prototype polluted via __proto__!");
console.log("✓ Prototype pollution blocked (__proto__)");

// Test 3: Prototype Pollution via constructor.prototype
let target3 = {};
let maliciousPayload2 = JSON.parse('{"constructor": {"prototype": {"polluted2": true}}}');
safeMerge(target3, maliciousPayload2);
assert.strictEqual({}.polluted2, undefined, "Prototype polluted via constructor!");
console.log("✓ Prototype pollution blocked (constructor.prototype)");

// Test 4: Existing fields overwritten safely
let target4 = { category: "Electronics" };
let source4 = { category: "Books" };
let result4 = safeMerge(target4, source4);
assert.strictEqual(result4.category, "Books", "Failed to overwrite property safely");
console.log("✓ Overwriting existing property passed");

console.log("All tests passed securely!");
